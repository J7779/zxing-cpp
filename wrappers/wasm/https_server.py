import http.server
import ssl
import os
import sys

# Certificate and key paths
cert_dir = os.path.dirname(os.path.abspath(__file__))
certfile = os.path.join(cert_dir, 'server.pem')

# Generate self-signed certificate using Python
def generate_self_signed_cert():
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        
        # Generate key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.DNSName(u"*"),
                x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
            ]),
            critical=False,
        ).sign(key, hashes.SHA256(), default_backend())
        
        # Write to PEM file (combined cert + key)
        with open(certfile, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        print(f"Generated certificate: {certfile}")
        return True
    except ImportError:
        return False

# Try to generate cert with cryptography library
import ipaddress
if not os.path.exists(certfile):
    if not generate_self_signed_cert():
        print("Installing cryptography package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cryptography", "-q"])
        generate_self_signed_cert()

# Change to script directory to serve files
os.chdir(cert_dir)

# Custom handler with Cross-Origin Isolation headers for SharedArrayBuffer/multi-threading
class COIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for SharedArrayBuffer and WASM multi-threading
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

# Start HTTPS server with COI headers
server_address = ('0.0.0.0', 8080)
httpd = http.server.HTTPServer(server_address, COIHandler)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile)
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving HTTPS on https://0.0.0.0:{8080}/")
print("Cross-Origin Isolation ENABLED for WASM multi-threading")
print("Accept the certificate warning in your browser")
httpd.serve_forever()