"""
Asymmetric cryptographic signing and verification services.

Uses Ed25519 for digital signatures — preferred over ECDSA-SECP256R1 because:
  - Simpler key format (32-byte private key, 32-byte public key)
  - Deterministic signatures (no random nonce required)
  - Faster signing and verification
  - Cleaner academic demonstration of the integrity vs authenticity distinction

Security architecture:
  - Private key: loaded from ARTIFACT_SIGNING_PRIVATE_KEY env var (CI/production)
    OR from artifacts/keys/ (development only — NOT suitable for production)
  - Public key: can be freely distributed for verification
  - The runtime should never possess the private key in production;
    use a signing service or HSM instead.

Terminology (used throughout the platform):
  - SHA-256 → integrity  (was it modified?)
  - Ed25519 signature → authenticity (who signed it?)
  - Provenance → traceability (how was it produced?)
"""
import os
import base64
import logging
from pathlib import Path
from typing import Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

from src.agentic_ml.core.constants import ARTIFACTS_DIR

logger = logging.getLogger("agentic_ml.security.crypto")

KEYS_DIR = ARTIFACTS_DIR / "keys"


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate an Ed25519 keypair.

    Returns:
        Tuple[bytes, bytes]: (private_key_pem, public_key_pem)
    """
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def sign_bytes(data: bytes, private_key_pem: bytes) -> str:
    """
    Sign binary data using Ed25519 private key.

    Returns:
        str: Base64-encoded digital signature string.

    Security note:
        Ed25519 signatures are deterministic — the same key and data always
        produce the same signature (unlike ECDSA which uses a random nonce).
    """
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError(
            "Expected Ed25519 private key. "
            "If you have an ECDSA key from a previous version, regenerate the keypair."
        )
    signature = private_key.sign(data)
    return base64.b64encode(signature).decode("utf-8")


def verify_signature(data: bytes, signature_b64: str, public_key_pem: bytes) -> bool:
    """
    Verify an Ed25519 digital signature over binary data.

    Returns:
        bool: True if signature is cryptographically valid, False otherwise.

    Provides authenticity guarantee:
        - True  → data was signed by the holder of the corresponding private key
        - False → data was modified, signature was forged, or wrong key used
    """
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        if not isinstance(public_key, Ed25519PublicKey):
            logger.debug("Provided key is not an Ed25519PublicKey")
            return False
        raw_sig = base64.b64decode(signature_b64.strip())
        public_key.verify(raw_sig, data)
        return True
    except (InvalidSignature, ValueError, Exception) as exc:
        logger.debug("Signature verification rejected: %s", exc)
        return False


def get_or_create_signing_keys() -> Tuple[bytes, bytes]:
    """
    Retrieve the Ed25519 signing keypair.

    Priority:
      1. Environment variables (ARTIFACT_SIGNING_PRIVATE_KEY / ARTIFACT_SIGNING_PUBLIC_KEY)
         → Use in CI, staging, and production. Never commit keys to source control.
      2. Disk cache at artifacts/keys/ (development only)
         → Automatically generated on first run.
         → NOTE: Do NOT use disk-cached keys in production environments.
    """
    env_priv = os.environ.get("ARTIFACT_SIGNING_PRIVATE_KEY")
    env_pub = os.environ.get("ARTIFACT_SIGNING_PUBLIC_KEY")
    if env_priv and env_pub:
        logger.debug("Signing keys loaded from environment variables.")
        return env_priv.encode("utf-8"), env_pub.encode("utf-8")

    # Development fallback — disk-cached keys
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    priv_file = KEYS_DIR / "artifact_signing_private_ed25519.pem"
    pub_file = KEYS_DIR / "artifact_signing_public_ed25519.pem"

    # Also check for legacy ECDSA key files and skip them
    if priv_file.exists() and pub_file.exists():
        logger.debug("Signing keys loaded from disk cache (development mode).")
        return priv_file.read_bytes(), pub_file.read_bytes()

    priv_pem, pub_pem = generate_keypair()
    priv_file.write_bytes(priv_pem)
    pub_file.write_bytes(pub_pem)
    logger.info(
        "Generated new Ed25519 signing keypair at %s "
        "[DEV MODE — do not use in production]",
        KEYS_DIR,
    )
    return priv_pem, pub_pem
