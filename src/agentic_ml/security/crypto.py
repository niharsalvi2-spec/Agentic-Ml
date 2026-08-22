"""
Asymmetric cryptographic signing and verification services.
Uses ECDSA (SECP256R1) with SHA-256 for high-security digital signatures.
"""
import os
import base64
import logging
from pathlib import Path
from typing import Tuple, Optional

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

from src.agentic_ml.core.constants import ARTIFACTS_DIR

logger = logging.getLogger("agentic_ml.security.crypto")

KEYS_DIR = ARTIFACTS_DIR / "keys"



def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generate an ECDSA (SECP256R1) keypair.
    Returns:
        Tuple[bytes, bytes]: (private_key_pem, public_key_pem)
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
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
    Sign binary data using ECDSA-SHA256 private key.
    Returns:
        str: Base64-encoded digital signature string.
    """
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    if not isinstance(private_key, ec.EllipticCurvePrivateKey):
        raise TypeError("Expected ECDSA private key (EllipticCurvePrivateKey)")
    signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(signature).decode("utf-8")


def verify_signature(data: bytes, signature_b64: str, public_key_pem: bytes) -> bool:
    """
    Verify ECDSA-SHA256 digital signature over binary data using trusted public key.
    Returns:
        bool: True if signature is cryptographically valid, False otherwise.
    """
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        if not isinstance(public_key, ec.EllipticCurvePublicKey):
            logger.debug("Provided key is not an EllipticCurvePublicKey")
            return False
        raw_sig = base64.b64decode(signature_b64.strip())
        public_key.verify(raw_sig, data, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError, Exception) as exc:
        logger.debug("Signature verification rejected: %s", exc)
        return False



def get_or_create_signing_keys() -> Tuple[bytes, bytes]:
    """
    Retrieve system signing keypair from environment variables, disk, or generate new.
    """
    env_priv = os.environ.get("ARTIFACT_SIGNING_PRIVATE_KEY")
    env_pub = os.environ.get("ARTIFACT_SIGNING_PUBLIC_KEY")
    if env_priv and env_pub:
        return env_priv.encode("utf-8"), env_pub.encode("utf-8")

    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    priv_file = KEYS_DIR / "artifact_signing_private.pem"
    pub_file = KEYS_DIR / "artifact_signing_public.pem"

    if priv_file.exists() and pub_file.exists():
        return priv_file.read_bytes(), pub_file.read_bytes()

    priv_pem, pub_pem = generate_keypair()
    priv_file.write_bytes(priv_pem)
    pub_file.write_bytes(pub_pem)
    logger.info("Generated new ECDSA signing keypair at %s", KEYS_DIR)
    return priv_pem, pub_pem

