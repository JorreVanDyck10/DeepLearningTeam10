"use strict";

function resolveApiUrl(location) {
  const netlifyFrontend = location.hostname === "10messi10.netlify.app";
  const separateLocalFrontend = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
    && location.port === "5173";
  const defaultUrl = netlifyFrontend
    ? "https://deeplearningteam10-api.onrender.com"
    : separateLocalFrontend
      ? `${location.protocol}//${location.hostname}:8000`
      : location.origin;

  return defaultUrl;
}

if (typeof module === "object" && module.exports) {
  module.exports = { resolveApiUrl };
} else {
  window.Team10APIConfig = { resolveApiUrl };
  window.GREENLAB_API_URL = resolveApiUrl(window.location);
}
