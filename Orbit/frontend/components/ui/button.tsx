import { cn } from "@/lib/utils";
import { forwardRef } from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "ghost" | "destructive" | "success";
  size?: "sm" | "md" | "lg";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-offset-1 disabled:opacity-50 disabled:pointer-events-none",
        size === "sm" && "px-3 py-1.5 text-sm",
        size === "md" && "px-4 py-2 text-sm",
        size === "lg" && "px-5 py-2.5 text-base",
        variant === "default" && "bg-gray-900 text-white hover:bg-gray-700 focus:ring-gray-500",
        variant === "outline" && "border border-gray-300 text-gray-700 hover:bg-gray-50 focus:ring-gray-300",
        variant === "ghost" && "text-gray-600 hover:bg-gray-100 focus:ring-gray-300",
        variant === "destructive" && "bg-red-600 text-white hover:bg-red-700 focus:ring-red-500",
        variant === "success" && "bg-green-600 text-white hover:bg-green-700 focus:ring-green-500",
        className
      )}
      {...props}
    />
  )
);
Button.displayName = "Button";
