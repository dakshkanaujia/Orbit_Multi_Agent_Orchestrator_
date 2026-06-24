import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";
import { Zap } from "lucide-react";
import { NavLinks } from "@/components/NavLinks";

export const metadata: Metadata = {
  title: "Orbit AI",
  description: "Multi-agent personal chief-of-staff",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-gray-50">
          <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-14">
                <Link href="/dashboard" className="flex items-center gap-2 font-bold text-gray-900 text-lg">
                  <Zap className="w-5 h-5 text-indigo-600" />
                  <span>Orbit AI</span>
                </Link>
                <NavLinks />
              </div>
            </div>
          </nav>
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
