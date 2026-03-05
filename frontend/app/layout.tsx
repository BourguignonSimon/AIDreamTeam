import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI Dream Team — React Sandbox',
  description: 'Stream-generate and preview React components in an isolated sandbox.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-full antialiased">{children}</body>
    </html>
  );
}
