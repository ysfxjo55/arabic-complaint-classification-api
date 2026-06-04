import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Complaint Analyst UI",
  description: "LibreChat-inspired UI for Arabic complaint classification and explainable routing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ar" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
