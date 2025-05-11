# LLM（Transformer）最小工作流矩阵计算完整例子（含LayerNorm与残差）

---

## 1. 输入嵌入 + 位置编码

假设有2个词元，词表大小为4，嵌入维度为2：

- 词元ID：[1, 2]
- 词嵌入矩阵（Embedding）：
  \[
  E = \begin{bmatrix}
  0.1 & 0.2 \\
  0.3 & 0.4 \\
  0.5 & 0.6 \\
  0.7 & 0.8
  \end{bmatrix}
  \]
- 位置编码（Position Embedding）：
  \[
  P = \begin{bmatrix}
  0.01 & 0.02 \\
  0.03 & 0.04
  \end{bmatrix}
  \]

- 输入嵌入：
  \[
  X = \begin{bmatrix}
  E[1] + P[0] \\
  E[2] + P[1]
  \end{bmatrix}
  = \begin{bmatrix}
  0.3+0.01 & 0.4+0.02 \\
  0.5+0.03 & 0.6+0.04
  \end{bmatrix}
  = \begin{bmatrix}
  0.31 & 0.42 \\
  0.53 & 0.64
  \end{bmatrix}
  \]

---

## 2. 进入Transformer前的LayerNorm

假设 LayerNorm 是对每一行做均值为0、方差为1的归一化（为简化，假设LN输出等于输入，实际会有缩放和偏置参数，这里忽略）。

\[
X_{LN0} = LN(X) \approx X
\]

---

## 3. Q/K/V 线性变换

假设 Q/K/V 的权重矩阵如下（无偏置）：

- \( W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \)
- \( W_K = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5 \end{bmatrix} \)
- \( W_V = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \)

则：
\[
Q = X_{LN0} \cdot W_Q = X_{LN0}
\]
\[
K = X_{LN0} \cdot W_K
= \begin{bmatrix}
0.31 \times 0.5 + 0.42 \times 0 & 0.31 \times 0 + 0.42 \times 0.5 \\
0.53 \times 0.5 + 0.64 \times 0 & 0.53 \times 0 + 0.64 \times 0.5
\end{bmatrix}
= \begin{bmatrix}
0.155 & 0.21 \\
0.265 & 0.32
\end{bmatrix}
\]
\[
V = X_{LN0} \cdot W_V
= \begin{bmatrix}
0.31 \times 1 + 0.42 \times 1 & 0.31 \times 1 + 0.42 \times 1 \\
0.53 \times 1 + 0.64 \times 1 & 0.53 \times 1 + 0.64 \times 1
\end{bmatrix}
= \begin{bmatrix}
0.73 & 0.73 \\
1.17 & 1.17
\end{bmatrix}
\]

---

## 4. 计算注意力分数

\[
\text{Attention Scores} = Q \cdot K^T
\]
\[
= \begin{bmatrix}
0.31 & 0.42 \\
0.53 & 0.64
\end{bmatrix}
\cdot
\begin{bmatrix}
0.155 & 0.265 \\
0.21 & 0.32
\end{bmatrix}^T
=
\begin{bmatrix}
0.31 & 0.42 \\
0.53 & 0.64
\end{bmatrix}
\cdot
\begin{bmatrix}
0.155 & 0.21 \\
0.265 & 0.32
\end{bmatrix}
\]
计算：
- 第一行第一列：0.31×0.155 + 0.42×0.21 = 0.04805 + 0.0882 = 0.13625
- 第一行第二列：0.31×0.265 + 0.42×0.32 = 0.08215 + 0.1344 = 0.21655
- 第二行第一列：0.53×0.155 + 0.64×0.21 = 0.08215 + 0.1344 = 0.21655
- 第二行第二列：0.53×0.265 + 0.64×0.32 = 0.14045 + 0.2048 = 0.34525

\[
\text{Attention Scores} =
\begin{bmatrix}
0.13625 & 0.21655 \\
0.21655 & 0.34525
\end{bmatrix}
\]

---

## 5. 缩放、Softmax

假设 d_k=2，缩放后：
\[
\text{Scaled} = \frac{\text{Attention Scores}}{\sqrt{2}}
\approx
\begin{bmatrix}
0.096 & 0.153 \\
0.153 & 0.244
\end{bmatrix}
\]

对每一行做 softmax（略去详细计算，结果近似）：

- 第一行 softmax: (0.49, 0.51)
- 第二行 softmax: (0.48, 0.52)

\[
\text{Softmax结果} =
\begin{bmatrix}
0.49 & 0.51 \\
0.48 & 0.52
\end{bmatrix}
\]

---

## 6. 加权求和（注意力输出）

\[
\text{Attention Output} = \text{Softmax} \cdot V
\]
- 第一行：0.49×[0.73,0.73] + 0.51×[1.17,1.17] = [0.3577+0.5967, 0.3577+0.5967] ≈ [0.954, 0.954]
- 第二行：0.48×[0.73,0.73] + 0.52×[1.17,1.17] = [0.3504+0.6084, 0.3504+0.6084] ≈ [0.959, 0.959]

\[
A = \begin{bmatrix}
0.954 & 0.954 \\
0.959 & 0.959
\end{bmatrix}
\]

---

## 7. MHA后的残差连接与LayerNorm

- 残差连接：将 MHA 输出与 MHA 之前的输入相加
- 再做 LayerNorm

\[
A_{res} = X_{LN0} + A = 
\begin{bmatrix}
0.31+0.954 & 0.42+0.954 \\
0.53+0.959 & 0.64+0.959
\end{bmatrix}
=
\begin{bmatrix}
1.264 & 1.374 \\
1.489 & 1.599
\end{bmatrix}
\]

\[
A_{LN1} = LN(A_{res}) \approx A_{res}
\]

---

## 8. 前馈网络（FFN）

假设 FFN 是一个线性层（权重全1，无偏置）：

\[
F = A_{LN1} \cdot W_{FFN}
\]
\[
W_{FFN} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
\]
\[
F = 
\begin{bmatrix}
1.264+1.374 & 1.264+1.374 \\
1.489+1.599 & 1.489+1.599
\end{bmatrix}
=
\begin{bmatrix}
2.638 & 2.638 \\
3.088 & 3.088
\end{bmatrix}
\]

---

## 9. FFN后的残差连接与LayerNorm

- 残差连接：FFN输出加上FFN前的输入（即A_{LN1}）
- 再做 LayerNorm

\[
F_{res} = A_{LN1} + F =
\begin{bmatrix}
1.264+2.638 & 1.374+2.638 \\
1.489+3.088 & 1.599+3.088
\end{bmatrix}
=
\begin{bmatrix}
3.902 & 4.012 \\
4.577 & 4.687
\end{bmatrix}
\]

\[
F_{LN2} = LN(F_{res}) \approx F_{res}
\]

---

## 10. 输出层 + Softmax

假设输出层是线性变换（权重全1，无偏置），输出 logits：

\[
\text{Logits} = F_{LN2} \cdot W_{out}
\]
\[
W_{out} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
\]
\[
\text{Logits} =
\begin{bmatrix}
3.902+4.012 & 3.902+4.012 \\
4.577+4.687 & 4.577+4.687
\end{bmatrix}
=
\begin{bmatrix}
7.914 & 7.914 \\
9.264 & 9.264
\end{bmatrix}
\]

对每一行做 softmax，输出概率分布（每行都为0.5, 0.5）。

---

## 总结

- 每个子层（MHA、FFN）前有 LayerNorm
- 每个子层后有残差连接（输出加输入）
- 这样能保证训练稳定性和信息流动 