import type { Completeness } from "../types";

export default function CompletenessFields({
  value,
  busy,
  onChange,
}: {
  value: Completeness;
  busy: boolean;
  onChange: (key: keyof Completeness, value: boolean) => void;
}) {
  return (
    <div className="completeness-list">
      {(["holdings", "trades", "guarantees", "registry"] as const).map(
        (key) => (
          <label key={key}>
            <span>
              {
                {
                  holdings: "持股记录",
                  trades: "交易记录",
                  guarantees: "担保记录",
                  registry: "登记与测度资料",
                }[key]
              }
            </span>
            <input
              type="checkbox"
              role="switch"
              checked={value[key]}
              disabled={busy}
              onChange={(event) => onChange(key, event.target.checked)}
            />
          </label>
        ),
      )}
    </div>
  );
}
