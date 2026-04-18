import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroTalk Voice Console",
  description: "Prototype UI for an advanced voice agent control surface.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
