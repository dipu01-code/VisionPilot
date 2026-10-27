import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionPilot — Agentic Computer Vision with OpenCV 5",
  description:
    "VisionPilot implements a perception-decision-action loop where OpenCV 5 analysis results drive agent decisions, tool selection, and human approval requests.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-black text-white antialiased">
        {children}
      </body>
    </html>
  );
}