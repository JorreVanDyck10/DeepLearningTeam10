// Render serves the website and API together: the API URL follows this deployment.
// The separate local frontend server on port 5173 uses the local API on port 8000.
const separateLocalFrontend = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname)
  && window.location.port === "5173";
window.GREENLAB_API_URL = separateLocalFrontend
  ? `${window.location.protocol}//${window.location.hostname}:8000`
  : window.location.origin;
