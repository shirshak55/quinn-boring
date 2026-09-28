use crate::alpn::AlpnProtocols;
use crate::bffi_ext::QuicSslContext;
use crate::error::{map_result, Result};
use crate::session_state::{SessionState, QUIC_METHOD};
use crate::version::QuicVersion;
use crate::{
    Entry, Error, KeyLog, NoKeyLog, NoSessionCache, QuicSsl, QuicSslSession, SessionCache,
    SimpleCache,
};
use btls::ssl::{Ssl, SslContext, SslContextBuilder, SslMethod, SslSession, SslVersion};
use btls_sys as bffi;
use bytes::{BufMut, Bytes, BytesMut};
use foreign_types_shared::ForeignType;
use once_cell::sync::Lazy;
use quinn_proto::{
    crypto, transport_parameters::TransportParameters, ConnectError, ConnectionId, Side,
    TransportError,
};
use std::any::Any;
use std::ffi::c_int;
use std::io::Cursor;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};
use tracing::{trace, warn};

/// Configuration for a client-side QUIC. Wraps around a BoringSSL [SslContext].
pub struct Config {
    ctx: SslContext,
    session_cache: Arc<dyn SessionCache>,
    key_log: Option<Arc<dyn KeyLog>>,
}

impl Config {
    pub fn new() -> Result<Self> {
        let mut builder = SslContextBuilder::new(SslMethod::tls())?;

        // QUIC requires TLS 1.3.
        builder.set_min_proto_version(Some(SslVersion::TLS1_3))?;
        builder.set_max_proto_version(Some(SslVersion::TLS1_3))?;

        builder.set_default_verify_paths()?;

        // We build the context early, since we are not allowed to further mutate the context
        // in start_session.
        let mut ctx = builder.build();

        // By default, enable early data (used for 0-RTT).
        ctx.enable_early_data(true);

        // Set the default ALPN protocols offered by the client. QUIC requires ALPN be configured
        // (see <https://www.rfc-editor.org/rfc/rfc9001.html#section-8.1>).
        ctx.set_alpn_protos(&AlpnProtocols::default().encode())?;

        // Configure session caching.
        ctx.set_session_cache_mode(bffi::SSL_SESS_CACHE_CLIENT | bffi::SSL_SESS_CACHE_NO_INTERNAL);
        ctx.set_new_session_callback(Some(Session::new_session_callback));

        // Set callbacks for the SessionState.
        ctx.set_quic_method(&QUIC_METHOD)?;
        ctx.set_info_callback(Some(SessionState::info_callback));

        // For clients, verification of the server is on by default.
        ctx.verify_peer(true);

        Ok(Self {
            ctx,
            session_cache: Arc::new(SimpleCache::new(256)),
            key_log: None,
        })
    }

    /// Returns the underlying [SslContext] backing all created sessions.
    pub fn ctx(&self) -> &SslContext {
        &self.ctx
    }

    /// Returns the underlying [SslContext] backing all created sessions. Wherever possible use
    /// the provided methods to modify settings rather than accessing this directly.
    ///
    /// Care should be taken to avoid overriding required behavior. In particular, this
    /// configuration will set callbacks for QUIC events, alpn selection, server name,
    /// as well as info and key logging.
    pub fn ctx_mut(&mut self) -> &mut SslContext {
        &mut self.ctx
    }

    /// Sets whether or not the peer certificate should be verified. If `true`, any error
    /// during verification will be fatal. If not called, verification of the server is
    /// enabled by default.
    pub fn verify_peer(&mut self, verify: bool) {
        self.ctx.verify_peer(verify)
    }

    /// Gets the [SessionCache] used to cache all client sessions.
    pub fn get_session_cache(&self) -> Arc<dyn SessionCache> {
        self.session_cache.clone()
    }

    /// Sets the [SessionCache] to be shared by all created client sessions.
    pub fn set_session_cache(&mut self, session_cache: Arc<dyn SessionCache>) {
        self.session_cache = session_cache;
    }

    /// Sets the ALPN protocols supported by the client. QUIC requires that
    /// ALPN be used (see <https://www.rfc-editor.org/rfc/rfc9001.html#section-8.1>).
    /// By default, the client will offer "h3".
    pub fn set_alpn(&mut self, alpn_protocols: &[Vec<u8>]) -> Result<()> {
        self.ctx
            .set_alpn_protos(&AlpnProtocols::from(alpn_protocols).encode())?;
        Ok(())
    }

    /// Sets the [KeyLog] for the client. By default, no key logging will occur.
    pub fn set_key_log(&mut self, key_log: Option<Arc<dyn KeyLog>>) {
        self.key_log = key_log;

        // Optimization for key logging. Only set the callback if a logger was supplied,
        // since the BoringSSL processing isn't free.
        match &self.key_log {
            Some(_) => {
                self.ctx
                    .set_keylog_callback(Some(SessionState::keylog_callback));
            }
            None => {
                self.ctx.set_keylog_callback(None);
            }
        }
    }
}

impl crypto::ClientConfig for Config {
    fn start_session(
        self: Arc<Self>,
        version: u32,
        server_name: &str,
        params: &TransportParameters,
    ) -> StdResult<Box<dyn crypto::Session>, ConnectError> {
        let version = QuicVersion::parse(version).unwrap();

        Ok(Session::new(self, version, server_name, params)
            .map_err(|_| ConnectError::EndpointStopping)?)
    }
}

static SESSION_INDEX: Lazy<c_int> = Lazy::new(|| unsafe {
    bffi::SSL_get_ex_new_index(0, std::ptr::null_mut(), std::ptr::null_mut(), None, None)
});

/// The [crypto::Session] implementation for BoringSSL.
struct Session {
    state: Box<SessionState>,
    server_name: Bytes,
    session_cache: Arc<dyn SessionCache>,
    zero_rtt_peer_params: Option<TransportParameters>,
    handshake_data_available: bool,
    handshake_data_sent: bool,
}

impl Session {
    fn new(
        cfg: Arc<Config>,
        version: QuicVersion,
        server_name: &str,
        params: &TransportParameters,
    ) -> Result<Box<Self>> {
        let session_cache = cfg.session_cache.clone();
        let mut ssl = Ssl::new(&cfg.ctx).unwrap();

        // Configure the TLS extension based on the QUIC version used.
        ssl.set_quic_use_legacy_codepoint(version.uses_legacy_extension());

        // Configure the SSL to be a client.
        ssl.set_connect_state();

        // Configure verification for the server hostname.
        ssl.set_verify_hostname(server_name)
            .map_err(|_| ConnectError::InvalidServerName(server_name.into()))?;

        // Set the SNI hostname.
        // TODO: should we validate the hostname?
        ssl.set_hostname(server_name)
            .map_err(|_| ConnectError::InvalidServerName(server_name.into()))?;

        // Set the transport parameters.
        ssl.set_quic_transport_params(&encode_params(params))
            .map_err(|_| ConnectError::EndpointStopping)?;

        let server_name_bytes = Bytes::copy_from_slice(server_name.as_bytes());

        // If we have a cached session, use it.
        let mut zero_rtt_peer_params = None;
        if let Some(entry) = session_cache.get(server_name_bytes.clone()) {
            match Entry::decode(ssl.ssl_context(), entry) {
                Ok(entry) => {
                    zero_rtt_peer_params = Some(entry.params);
                    match unsafe { ssl.set_session(entry.session.as_ref()) } {
                        Ok(()) => {
                            trace!("attempting resumption (0-RTT) for server: {}.", server_name);
                        }
                        Err(e) => {
                            warn!(
                                "failed setting cached session for server {}: {:?}",
                                server_name, e
                            )
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "failed decoding session entry for server {}: {:?}",
                        server_name, e
                    )
                }
            }
        } else {
            trace!(
                "no cached session found for server: {}. Will continue with 1-RTT.",
                server_name
            );
        }

        Self::start(
            ssl,
            version,
            server_name_bytes,
            session_cache,
            zero_rtt_peer_params,
            cfg.key_log
                .as_ref()
                .map_or(Arc::new(NoKeyLog), |key_log| key_log.clone()),
        )
    }

    /// A session driving `ssl` as [`MirrorConfig`] describes.
    fn mirrored(
        mut ssl: Ssl,
        resumed_transport_parameters: Option<&[u8]>,
        early_data: bool,
        version: QuicVersion,
        server_name: &str,
        params: &[u8],
    ) -> Result<Box<Self>> {
        ssl.set_quic_method(&QUIC_METHOD)?;
        ssl.set_quic_use_legacy_codepoint(version.uses_legacy_extension());
        ssl.set_connect_state();
        ssl.set_quic_transport_params(params)?;

        let zero_rtt_peer_params = resumed_transport_parameters
            .map(|mut params| TransportParameters::read(Side::Client, &mut params))
            .transpose()
            .map_err(|e| {
                Error::invalid_input(format!(
                    "failed parsing resumed transport parameters: {:?}",
                    e
                ))
            })?;
        // quinn needs the server's transport parameters to send 0-RTT data.
        ssl.set_early_data_enabled(early_data && zero_rtt_peer_params.is_some());

        Self::start(
            ssl,
            version,
            Bytes::copy_from_slice(server_name.as_bytes()),
            Arc::new(NoSessionCache),
            zero_rtt_peer_params,
            Arc::new(NoKeyLog),
        )
    }

    /// Starts the handshake of a session driving `ssl`, configured as a client.
    fn start(
        ssl: Ssl,
        version: QuicVersion,
        server_name: Bytes,
        session_cache: Arc<dyn SessionCache>,
        zero_rtt_peer_params: Option<TransportParameters>,
        key_log: Arc<dyn KeyLog>,
    ) -> Result<Box<Self>> {
        let mut session = Box::new(Self {
            state: SessionState::new(ssl, Side::Client, version, key_log)?,
            server_name,
            session_cache,
            zero_rtt_peer_params,
            handshake_data_available: false,
            handshake_data_sent: false,
        });

        // Register the instance in SSL ex_data. This allows the static callbacks to
        // reference the instance.
        unsafe {
            map_result(bffi::SSL_set_ex_data(
                session.state.ssl.as_ptr(),
                *SESSION_INDEX,
                &mut *session as *mut Self as *mut _,
            ))?;
        }

        // Start the handshake in order to emit the Client Hello on the first
        // call to write_handshake.
        session.state.advance_handshake()?;

        Ok(session)
    }

    /// Handler for the rejection of a 0-RTT attempt. Will continue with 1-RTT.
    fn on_zero_rtt_rejected(&mut self) {
        trace!(
            "0-RTT handshake attempted but was rejected by the server: {}",
            Ssl::early_data_reason_string(self.state.ssl.get_early_data_reason())
        );

        self.zero_rtt_peer_params = None;

        // Removed the failed cache entry.
        self.session_cache.remove(self.server_name.clone());

        // Now retry advancing the handshake, this time in 1-RTT mode.
        if let Err(e) = self.state.advance_handshake() {
            warn!("failed advancing 1-RTT handshake: {:?}", e)
        }
    }

    /// Client-side only callback from BoringSSL to allow caching of a new session.
    fn on_new_session(&mut self, session: SslSession) {
        if !session.early_data_capable() {
            warn!("failed caching session: not early data capable");
            return;
        }

        // Get the server transport parameters.
        let params = match self.state.ssl.get_peer_quic_transport_params() {
            Some(params) => {
                match TransportParameters::read(Side::Client, &mut Cursor::new(&params)) {
                    Ok(params) => params,
                    Err(e) => {
                        warn!("failed parsing server transport parameters: {:?}", e);
                        return;
                    }
                }
            }
            None => {
                warn!("failed caching session: server transport parameters are not available");
                return;
            }
        };

        // Encode the session cache entry, including both the session and the server params.
        let entry = Entry { session, params };
        match entry.encode() {
            Ok(value) => {
                // Cache the session.
                self.session_cache.put(self.server_name.clone(), value)
            }
            Err(e) => {
                warn!("failed caching session: unable to encode entry: {:?}", e);
            }
        }
    }

    /// Called by the static callbacks to retrieve the instance pointer.
    #[inline]
    fn get_instance(ssl: *const bffi::SSL) -> &'static mut Session {
        unsafe {
            let data = bffi::SSL_get_ex_data(ssl, *SESSION_INDEX);
            if data.is_null() {
                panic!("BUG: Session instance missing")
            }
            &mut *(data as *mut Session)
        }
    }

    /// Raw callback from BoringSSL.
    extern "C" fn new_session_callback(
        ssl: *mut bffi::SSL,
        session: *mut bffi::SSL_SESSION,
    ) -> c_int {
        let inst = Self::get_instance(ssl);
        let session = unsafe { SslSession::from_ptr(session) };
        inst.on_new_session(session);

        // Return 1 to indicate we've taken ownership of the session.
        1
    }
}

impl crypto::Session for Session {
    fn initial_keys(&self, dcid: &ConnectionId, side: Side) -> crypto::Keys {
        self.state.initial_keys(dcid, side)
    }

    fn handshake_data(&self) -> Option<Box<dyn Any>> {
        self.state.handshake_data()
    }

    fn peer_identity(&self) -> Option<Box<dyn Any>> {
        self.state.peer_identity()
    }

    fn early_crypto(&self) -> Option<(Box<dyn crypto::HeaderKey>, Box<dyn crypto::PacketKey>)> {
        self.state.early_crypto()
    }

    fn early_data_accepted(&self) -> Option<bool> {
        Some(self.state.ssl.early_data_accepted())
    }

    fn is_handshaking(&self) -> bool {
        self.state.is_handshaking()
    }

    fn read_handshake(&mut self, plaintext: &[u8]) -> StdResult<bool, TransportError> {
        self.state.read_handshake(plaintext)?;

        if self.state.early_data_rejected {
            self.on_zero_rtt_rejected();
        }

        // Only indicate that handshake data is available once.
        // On the client side there is no ALPN callback, so we need to manually check
        // if the ALPN protocol has been selected.
        if !self.handshake_data_sent {
            if self.state.ssl.selected_alpn_protocol().is_some() {
                self.handshake_data_available = true;
            }

            if self.handshake_data_available {
                self.handshake_data_sent = true;
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn transport_parameters(&self) -> StdResult<Option<TransportParameters>, TransportError> {
        match self.state.transport_parameters()? {
            Some(params) => Ok(Some(params)),
            None => {
                if self.state.ssl.in_early_data() {
                    Ok(self.zero_rtt_peer_params)
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn write_handshake(&mut self, buf: &mut Vec<u8>) -> Option<crypto::Keys> {
        self.state.write_handshake(buf)
    }

    fn next_1rtt_keys(&mut self) -> Option<crypto::KeyPair<Box<dyn crypto::PacketKey>>> {
        self.state.next_1rtt_keys()
    }

    fn is_valid_retry(&self, orig_dst_cid: &ConnectionId, header: &[u8], payload: &[u8]) -> bool {
        self.state.is_valid_retry(orig_dst_cid, header, payload)
    }

    fn export_keying_material(
        &self,
        output: &mut [u8],
        label: &[u8],
        context: &[u8],
    ) -> StdResult<(), crypto::ExportKeyingMaterialError> {
        self.state.export_keying_material(output, label, context)
    }
}

fn encode_params(params: &TransportParameters) -> Bytes {
    let mut out = BytesMut::with_capacity(128);
    params.write(&mut out);
    out.freeze()
}

/// A client configuration for one connection that starts from a caller's [`Ssl`] and sends a
/// caller's transport parameters, so the connection offers another client's ClientHello and
/// transport parameters.
///
/// The [`Ssl`] already verifies and announces (SNI) the server name as it should, offers the
/// ClientHello, and holds the session to resume, if any; this adds the QUIC transport. Its
/// context's callbacks are the caller's: this installs none, so a caller keeps the sessions the
/// connection receives with its context's new-session callback (with the server's transport
/// parameters, `SSL_get_peer_quic_transport_params`, for 0-RTT). A configuration serves a
/// single connection.
pub struct MirrorConfig {
    ssl: Mutex<Option<Ssl>>,
    transport_parameters: Vec<(u64, Bytes)>,
    resumed_transport_parameters: Option<Bytes>,
    early_data: bool,
}

impl MirrorConfig {
    /// A connection starting from `ssl` and sending `transport_parameters` as listed, in order,
    /// with its own initial_source_connection_id in place of the listed one's value.
    pub fn new(ssl: Ssl, transport_parameters: Vec<(u64, Bytes)>) -> Self {
        Self {
            ssl: Mutex::new(Some(ssl)),
            transport_parameters,
            resumed_transport_parameters: None,
            early_data: false,
        }
    }

    /// Sets the transport parameters the server sent on the connection that received the
    /// session the [`Ssl`] resumes, encoded as it sent them. With them, the connection can
    /// send 0-RTT data (see [`MirrorConfig::set_early_data`]).
    pub fn set_resumed_transport_parameters(&mut self, params: Option<Bytes>) {
        self.resumed_transport_parameters = params;
    }

    /// Sets whether the connection offers 0-RTT (early data) when it resumes a session that
    /// allows it and has the server's transport parameters. Off by default.
    pub fn set_early_data(&mut self, enabled: bool) {
        self.early_data = enabled;
    }

    /// `own`, quinn's transport parameters for the connection, as the listed ones.
    fn encode(&self, own: &TransportParameters) -> StdResult<Bytes, ConnectError> {
        const INITIAL_SOURCE_CONNECTION_ID: u64 = 0x0f;
        let own = encode_params(own);
        let mut reader = own.as_ref();
        let mut scid = None;
        while !reader.is_empty() {
            let id = read_varint(&mut reader).ok_or(ConnectError::EndpointStopping)?;
            let len = read_varint(&mut reader).ok_or(ConnectError::EndpointStopping)? as usize;
            let value = reader.get(..len).ok_or(ConnectError::EndpointStopping)?;
            if id == INITIAL_SOURCE_CONNECTION_ID {
                scid = Some(value);
            }
            reader = &reader[len..];
        }
        let scid = scid.ok_or(ConnectError::EndpointStopping)?;
        let mut out = BytesMut::new();
        for (id, value) in &self.transport_parameters {
            let value = match *id {
                INITIAL_SOURCE_CONNECTION_ID => scid,
                _ => value,
            };
            write_varint(&mut out, *id);
            write_varint(&mut out, value.len() as u64);
            out.extend_from_slice(value);
        }
        Ok(out.freeze())
    }
}

impl crypto::ClientConfig for MirrorConfig {
    fn start_session(
        self: Arc<Self>,
        version: u32,
        server_name: &str,
        params: &TransportParameters,
    ) -> StdResult<Box<dyn crypto::Session>, ConnectError> {
        let version = QuicVersion::parse(version).map_err(|_| ConnectError::UnsupportedVersion)?;
        let params = self.encode(params)?;
        let ssl = self
            .ssl
            .lock()
            .unwrap()
            .take()
            .ok_or(ConnectError::EndpointStopping)?;
        Ok(Session::mirrored(
            ssl,
            self.resumed_transport_parameters.as_deref(),
            self.early_data,
            version,
            server_name,
            &params,
        )
        .map_err(|e| {
            warn!(
                "failed starting mirrored session for {}: {:?}",
                server_name, e
            );
            ConnectError::EndpointStopping
        })?)
    }
}

/// Reads a QUIC variable-length integer.
fn read_varint(buf: &mut &[u8]) -> Option<u64> {
    let first = *buf.first()?;
    let len = 1usize << (first >> 6);
    let bytes = buf.get(..len)?;
    let value = bytes[1..]
        .iter()
        .fold(u64::from(first & 0x3f), |value, byte| {
            value << 8 | u64::from(*byte)
        });
    *buf = &buf[len..];
    Some(value)
}

/// Writes a QUIC variable-length integer.
fn write_varint(out: &mut BytesMut, value: u64) {
    match value {
        0..=0x3f => out.put_u8(value as u8),
        0x40..=0x3fff => out.put_u16(value as u16 | 0x4000),
        0x4000..=0x3fff_ffff => out.put_u32(value as u32 | 0x8000_0000),
        _ => out.put_u64(value | 0xc000_0000_0000_0000),
    }
}
