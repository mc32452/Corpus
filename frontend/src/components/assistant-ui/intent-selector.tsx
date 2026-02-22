"use client";

import { memo, useEffect, type ComponentPropsWithoutRef } from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { type VariantProps } from "class-variance-authority";
import { CheckIcon } from "lucide-react";
import { useAssistantApi } from "@assistant-ui/react";
import { cn } from "@/lib/utils";
import {
  SelectRoot,
  SelectTrigger,
  SelectContent,
  selectTriggerVariants,
} from "@/components/ui/select";
import { useAppDispatch, useAppState } from "@/context/app-context";

// ---------------------------------------------------------------------------
// Intent options
// ---------------------------------------------------------------------------

export interface IntentOption {
  id: string;
  name: string;
  description: string;
}

export const INTENT_OPTIONS: IntentOption[] = [
  {
    id: "auto",
    name: "Auto",
    description: "The system automatically chooses the best response mode for your question.",
  },
  {
    id: "summarise",
    name: "Summarise",
    description: "Pulls out the key points and main ideas concisely.",
  },
  {
    id: "explain",
    name: "Explain",
    description: "Breaks down complex or technical content into plain language.",
  },
  {
    id: "analyze",
    name: "Analyze",
    description: "Examines deeper meanings, themes, and significance in the text.",
  },
  {
    id: "compare",
    name: "Compare",
    description: "Shows similarities and differences between multiple ideas or documents.",
  },
  {
    id: "critique",
    name: "Critique",
    description: "Highlights the strengths, weaknesses, and limitations in the text.",
  },
  {
    id: "factual",
    name: "Factual",
    description: "Provides a direct, specific answer to a concrete question.",
  },
  {
    id: "collection",
    name: "Collection",
    description: "Describes the overall themes and scope of your full document set.",
  },
  {
    id: "extract",
    name: "Extract",
    description: "Pulls out specific data like names, dates, or figures into a list.",
  },
  {
    id: "timeline",
    name: "Timeline",
    description: "Organizes events in chronological order.",
  },
  {
    id: "how_to",
    name: "How-To",
    description: "Presents clear, step-by-step instructions from the document.",
  },
  {
    id: "quote_evidence",
    name: "Quote / Evidence",
    description: "Returns relevant direct quotes to support a claim or question.",
  },
];

// ---------------------------------------------------------------------------
// IntentSelectorItem
// ---------------------------------------------------------------------------

function IntentSelectorItem({ option }: { option: IntentOption }) {
  return (
    <SelectPrimitive.Item
      value={option.id}
      textValue={option.name}
      className={cn(
        "relative flex w-full cursor-default select-none items-start gap-2 rounded-lg py-2 pr-9 pl-3 text-sm outline-none",
        "focus:bg-accent focus:text-accent-foreground",
        "data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
      )}
    >
      <span className="absolute right-3 top-2.5 flex size-4 items-center justify-center">
        <SelectPrimitive.ItemIndicator>
          <CheckIcon className="size-4" />
        </SelectPrimitive.ItemIndicator>
      </span>
      <SelectPrimitive.ItemText>
        <span className="flex flex-col gap-0.5">
          <span className="font-medium leading-tight">{option.name}</span>
          <span className="text-muted-foreground text-xs leading-tight max-w-48">
            {option.description}
          </span>
        </span>
      </SelectPrimitive.ItemText>
    </SelectPrimitive.Item>
  );
}

// ---------------------------------------------------------------------------
// IntentSelector
// ---------------------------------------------------------------------------

export type IntentSelectorProps = VariantProps<typeof selectTriggerVariants> & {
  contentClassName?: string;
} & Omit<ComponentPropsWithoutRef<typeof SelectRoot>, "value" | "onValueChange">;

const IntentSelectorImpl = ({
  variant,
  size,
  contentClassName,
  ...forwardedProps
}: IntentSelectorProps) => {
  const dispatch = useAppDispatch();
  const { intentOverride } = useAppState();
  const api = useAssistantApi();

  // Register intent with the assistant runtime so it's read per-request
  // (same pattern as ModelSelector — avoids stale closure in body.data).
  // Cast needed because LanguageModelConfig doesn't expose custom fields;
  // the backend reads model_extra.config.intentOverride from the raw JSON.
  useEffect(() => {
    // Cast to `any` because LanguageModelConfig doesn't expose custom fields;
    // the backend reads model_extra.config.intentOverride from the raw JSON.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const config = { config: { intentOverride } as any };
    return api.modelContext().register({ getModelContext: () => config });
  }, [api, intentOverride]);

  const handleValueChange = (value: string) => {
    dispatch({ type: "SET_INTENT_OVERRIDE", intentOverride: value });
  };

  const currentOption = INTENT_OPTIONS.find((o) => o.id === intentOverride) ?? INTENT_OPTIONS[0];
  const isAuto = intentOverride === "auto";

  return (
    <SelectRoot
      value={intentOverride}
      onValueChange={handleValueChange}
      {...forwardedProps}
    >
      <SelectTrigger
        variant={variant}
        size={size}
        className={cn(
          "aui-intent-selector-trigger text-muted-foreground gap-1.5",
          !isAuto && "text-blue-400",
        )}
        aria-label="Select response mode"
      >
        {isAuto ? (
          <span className="font-medium">Auto</span>
        ) : (
          <span className="flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-blue-400 shrink-0" />
            <span className="font-medium">{currentOption.name}</span>
          </span>
        )}
      </SelectTrigger>

      <SelectContent
        className={cn("min-w-[220px] max-h-64 overflow-y-auto", contentClassName)}
      >
        {/* Auto option at the top, separated */}
        <IntentSelectorItem option={INTENT_OPTIONS[0]} />
        <div className="my-1 border-t border-border/50" />
        {INTENT_OPTIONS.slice(1).map((option) => (
          <IntentSelectorItem key={option.id} option={option} />
        ))}
      </SelectContent>
    </SelectRoot>
  );
};

export const IntentSelector = memo(IntentSelectorImpl);
IntentSelector.displayName = "IntentSelector";
