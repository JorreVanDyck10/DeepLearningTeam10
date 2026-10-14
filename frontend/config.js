"use strict";

function resolveApiUrl(location, savedUrl) {
  const netlifyFrontend = location.hostname === "10messi10.netlify.app";
  const separateLocalFrontend = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
    && location.port === "5173";
  const defaultUrl = netlifyFrontend
    ? "https://deeplearningteam10-api.onrender.com"
    : separateLocalFrontend
      ? `${location.protocol}//${location.hostname}:8000`
      : location.origin;

  // Migrate the old Netlify default; retain deliberately configured backends.
  if (savedUrl && !(netlifyFrontend && savedUrl.replace(/\/+$/, "") === location.origin)) {
    return savedUrl.replace(/\/+$/, "");
  }
  return defaultUrl;
}

if (typeof module === "object" && module.exports) {
  module.exports = { resolveApiUrl };
} else {
  window.Team10APIConfig = { resolveApiUrl };
  window.GREENLAB_API_URL = resolveApiUrl(window.location);
}
