export class ApiError extends Error {
  constructor(
    message: string,
    public code: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new ApiError(
      "无法连接本地引擎，请确认网站服务仍在运行。",
      "network",
      0,
    );
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      "服务器没有返回有效数据，请重试。",
      "invalid_response",
      response.status,
    );
  }
  if (!response.ok) {
    const value = data as { error?: { message?: string; code?: string } };
    throw new ApiError(
      value.error?.message || "操作未完成，请重试。",
      value.error?.code || "request_failed",
      response.status,
    );
  }
  return data as T;
}
export function canonical(value: unknown): string {
  const normalize = (item: unknown): unknown =>
    Array.isArray(item)
      ? item.map(normalize)
      : item && typeof item === "object"
        ? Object.fromEntries(
            Object.entries(item)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([k, v]) => [k, normalize(v)]),
          )
        : item;
  return JSON.stringify(normalize(value));
}
export function downloadJson(value: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], {
      type: "application/json;charset=utf-8",
    }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export const fmt = (value: number, digits = 3) =>
  value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
export function shortName(name: string): string {
  return name
    .replace(/股份有限公司|有限责任公司|有限公司|合伙企业|股份公司/g, "")
    .replace(/集团$/, "");
}
