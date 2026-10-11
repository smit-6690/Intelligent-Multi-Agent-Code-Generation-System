import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from magent_codegen.service import Handler


def test_health_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["status"] == "ok"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
