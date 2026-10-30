import type { RuleExample } from "../rule-notation";
import RuleExpression from "./RuleExpression";

export default function RuleNotation({
  example,
  dataReady = true,
}: {
  example: RuleExample | null;
  dataReady?: boolean;
}) {
  return (
    <aside className="rule-notation" aria-label="规则变量说明">
      <p className="rule-notation-intro">
        <strong>这里是规则模板，不是具体数据。</strong>
        <code>?</code>{" "}
        开头的符号表示待匹配的变量，不是数据缺失；同一条规则中的同名变量必须代入同一个值。
      </p>
      <p>
        <code>∧</code> 表示“并且”，所有前提和比较条件都要成立；
        <code>→</code> 表示在当前教学规则下可以推导结论。
      </p>
      <details className="rule-notation-details">
        <summary>变量含义与当前案例示例</summary>
        <dl className="rule-symbols">
          <div>
            <dt>
              <code>?x、?y、?p、?q、?c</code>
            </dt>
            <dd>实体变量；主体和目标由谓词中的位置决定。</dd>
          </div>
          <div>
            <dt>
              <code>?r、?r1、?r2</code>
            </dt>
            <dd>持股比例；0.8 表示 80%。</dd>
          </div>
          <div>
            <dt>
              <code>?m</code>
            </dt>
            <dd>金额变量，单位为万元。</dd>
          </div>
          <div>
            <dt>
              <code>?rel、?t</code>
            </dt>
            <dd>亲属关系名称或预警类型。</dd>
          </div>
        </dl>
        {example ? (
          <div
            className="rule-binding-example"
            aria-label="当前案例变量代入示例"
          >
            <p>
              <strong>实际记录</strong> <code>{example.fact}</code>
            </p>
            <p>{example.description}</p>
            <ul className="rule-bindings" aria-label="变量绑定">
              {example.bindings.map((binding) => (
                <li key={binding.variable}>
                  <code>
                    {binding.variable} = {binding.value}
                  </code>
                </li>
              ))}
            </ul>
            <p>
              <strong>代入 {example.rule} 后的完整规则</strong>
            </p>
            <RuleExpression
              antecedents={example.antecedents}
              conclusions={[example.consequent]}
            />
            <p className="rule-example-disclosure">
              这是变量代入示例，不是已运行的推理结果。是否触发仍需满足全部条件并运行推理，不能据此作现实控制认定。
            </p>
          </div>
        ) : (
          <p className="rule-example-empty">
            {dataReady
              ? "当前没有可用于演示的完整持股规则或记录，暂不能生成实际绑定示例。"
              : "当前案例尚未载入，载入后可查看实际记录的变量代入。"}
          </p>
        )}
      </details>
    </aside>
  );
}
