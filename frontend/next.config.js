/** @type {import('next').NextConfig} */
const nextConfig = {
  /**
   * Custom HTTP headers.
   *
   * /sandbox-shell.html is served from /public with a strict CSP so that:
   *  - React + ReactDOM + Babel can load from unpkg.com
   *  - User code inside the sandbox cannot make network requests (connect-src 'none')
   *  - No storage access (no cookies, localStorage – enforced by iframe sandbox attr too)
   */
  async headers() {
    return [
      {
        source: '/sandbox-shell.html',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'none'",
              "script-src 'unsafe-inline' https://unpkg.com",
              "style-src 'unsafe-inline'",
              "connect-src 'none'",
              "img-src data: blob:",
              "font-src 'none'",
              "object-src 'none'",
              "frame-src 'none'",
              "form-action 'none'",
              "base-uri 'none'",
            ].join('; '),
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'Cache-Control',
            value: 'no-store',
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
