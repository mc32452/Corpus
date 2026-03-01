import type { ComponentPropsWithoutRef, HTMLAttributes } from "react";
import {
  BracesIcon,
  DownloadIcon,
  FileIcon,
  FileTextIcon,
  ImageIcon,
  MusicIcon,
  VideoIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

type FileVariant = "outline" | "ghost" | "muted";
type FileSizeVariant = "sm" | "default" | "lg";

const rootVariantClasses: Record<FileVariant, string> = {
  outline: "border border-border bg-transparent",
  ghost: "border border-transparent bg-transparent",
  muted: "border border-border/50 bg-muted/40",
};

const rootSizeClasses: Record<FileSizeVariant, string> = {
  sm: "gap-2 px-2 py-1.5 text-xs",
  default: "gap-2.5 px-2.5 py-2 text-sm",
  lg: "gap-3 px-3 py-2.5 text-sm",
};

export function formatFileSize(bytes?: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}


type FileRootProps = HTMLAttributes<HTMLDivElement> & {
  variant?: FileVariant;
  size?: FileSizeVariant;
};

export function FileRoot({
  className,
  variant = "outline",
  size = "default",
  ...props
}: FileRootProps) {
  return (
    <div
      className={cn(
        "inline-flex min-w-0 items-center rounded-lg text-foreground",
        rootVariantClasses[variant],
        rootSizeClasses[size],
        className,
      )}
      {...props}
    />
  );
}

type FileIconDisplayProps = {
  mimeType?: string;
} & ComponentPropsWithoutRef<"span">;

export function FileIconDisplay({
  mimeType,
  className,
  children,
  ...props
}: FileIconDisplayProps) {
  const icon = (() => {
    if (!mimeType) return <FileIcon className="size-5" aria-hidden="true" />;
    if (mimeType.startsWith("image/")) {
      return <ImageIcon className="size-5" aria-hidden="true" />;
    }
    if (mimeType === "application/pdf") {
      return <FileTextIcon className="size-5" aria-hidden="true" />;
    }
    if (mimeType === "application/json") {
      return <BracesIcon className="size-5" aria-hidden="true" />;
    }
    if (mimeType.startsWith("text/")) {
      return <FileTextIcon className="size-5" aria-hidden="true" />;
    }
    if (mimeType.startsWith("audio/")) {
      return <MusicIcon className="size-5" aria-hidden="true" />;
    }
    if (mimeType.startsWith("video/")) {
      return <VideoIcon className="size-5" aria-hidden="true" />;
    }
    return <FileIcon className="size-5" aria-hidden="true" />;
  })();

  return (
    <span className={cn("shrink-0 text-muted-foreground", className)} {...props}>
      {children ?? icon}
    </span>
  );
}

export function FileName({ className, ...props }: ComponentPropsWithoutRef<"span">) {
  return <span className={cn("truncate font-medium", className)} {...props} />;
}

type FileSizeProps = {
  bytes?: number;
} & ComponentPropsWithoutRef<"span">;

export function FileSize({ bytes = 0, className, ...props }: FileSizeProps) {
  return (
    <span className={cn("text-xs text-muted-foreground", className)} {...props}>
      {formatFileSize(bytes)}
    </span>
  );
}

type FileDownloadProps = {
  data: string;
  mimeType?: string;
  filename: string;
} & Omit<ComponentPropsWithoutRef<"button">, "onClick">;

export function FileDownload({
  data,
  mimeType = "application/octet-stream",
  filename,
  className,
  ...props
}: FileDownloadProps) {
  const onClick = () => {
    const href = data.startsWith("data:") ? data : `data:${mimeType};base64,${data}`;
    const link = document.createElement("a");
    link.href = href;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground",
        className,
      )}
      aria-label={`Download ${filename}`}
      {...props}
    >
      <DownloadIcon className="size-4" aria-hidden="true" />
    </button>
  );
}

type FileProps = {
  name: string;
  mimeType?: string;
  bytes?: number;
  variant?: FileVariant;
  size?: FileSizeVariant;
  className?: string;
};

function FileBase({
  name,
  mimeType,
  bytes,
  variant = "outline",
  size = "default",
  className,
}: FileProps) {
  return (
    <FileRoot variant={variant} size={size} className={className}>
      <FileIconDisplay mimeType={mimeType} />
      <div className="min-w-0 flex-1">
        <FileName>{name}</FileName>
        {typeof bytes === "number" && <FileSize bytes={bytes} />}
      </div>
    </FileRoot>
  );
}

export const File = Object.assign(FileBase, {
  Root: FileRoot,
  Icon: FileIconDisplay,
  Name: FileName,
  Size: FileSize,
  Download: FileDownload,
});
