import { useEffect, useRef, useState } from "react";

type Props = {
  value: number;
  maximum: number;
  busy: boolean;
  onChange: (ratio: number) => void;
  onValidity: (valid: boolean) => void;
};

/** Keep unfinished numeric text local. Never run the last-good value behind an invalid field. */
export default function HoldingEditor({
  value,
  maximum,
  busy,
  onChange,
  onValidity,
}: Props) {
  const [text, setText] = useState(String(Number((value * 100).toFixed(2))));
  const [error, setError] = useState("");
  const lastApplied = useRef(value);
  useEffect(() => {
    if (value !== lastApplied.current) {
      setText(String(Number((value * 100).toFixed(2))));
      setError("");
      onValidity(true);
    }
    lastApplied.current = value;
  }, [value, onValidity]);
  useEffect(() => () => onValidity(true), [onValidity]);

  function update(raw: string) {
    setText(raw);
    const amount = Number(raw);
    const valid =
      raw.trim() !== "" &&
      Number.isFinite(amount) &&
      amount >= 0.01 &&
      amount <= maximum + 1e-8;
    setError(valid ? "" : `请输入 0.01–${maximum.toFixed(2)} 之间的百分数。`);
    onValidity(valid);
    if (valid) {
      lastApplied.current = amount / 100;
      onChange(amount / 100);
    }
  }

  return (
    <>
      <input
        id="holding-range"
        type="range"
        min="0.01"
        max={maximum}
        step="0.01"
        value={value * 100}
        disabled={busy}
        aria-label="持股比例滑块"
        onChange={(event) => update(event.target.value)}
      />
      <div className="number-field">
        <input
          id="holding-value"
          type="text"
          inputMode="decimal"
          value={text}
          disabled={busy}
          aria-invalid={!!error}
          aria-describedby={
            error ? "holding-error holding-help" : "holding-help"
          }
          onChange={(event) => update(event.target.value)}
          onBlur={() => {
            if (!error) setText(String(Number(text)));
          }}
        />
        <span>%</span>
      </div>
      {error && (
        <p id="holding-error" className="field-error" role="alert">
          {error} 修正后才能运行推理。
        </p>
      )}
    </>
  );
}
