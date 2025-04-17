import streamlit as st
import numpy as np
import pandas as pd

def to_row_echelon(matrix):
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    pivot_row = 0

    for col in range(cols):
        for r in range(pivot_row, rows):
            if A[r, col] != 0:
                A[[pivot_row, r]] = A[[r, pivot_row]]
                break
        else:
            continue

        A[pivot_row] = A[pivot_row] / A[pivot_row, col]

        for r in range(pivot_row + 1, rows):
            A[r] = A[r] - A[r, col] * A[pivot_row]

        pivot_row += 1
        if pivot_row == rows:
            break

    return A

def to_row_echelon_with_steps(matrix):
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    pivot_row = 0
    steps = [("初期行列", A.copy())]

    for col in range(cols):
        # ピボット探しと交換
        for r in range(pivot_row, rows):
            if A[r, col] != 0:
                if r != pivot_row:
                    A[[pivot_row, r]] = A[[r, pivot_row]]
                    steps.append((f"行 {pivot_row+1} と 行 {r+1} を交換", A.copy()))
                break
        else:
            continue

        # ピボットを1にする
        pivot = A[pivot_row, col]
        if pivot != 1:
            A[pivot_row] = A[pivot_row] / pivot
            steps.append((f"行 {pivot_row+1} を {pivot:.2f} で割る", A.copy()))

        # 下の行を0にする
        for r in range(pivot_row + 1, rows):
            factor = A[r, col]
            if factor != 0:
                A[r] = A[r] - factor * A[pivot_row]
                steps.append((f"行 {r+1} から {factor:.2f} × 行 {pivot_row+1} を引く", A.copy()))

        pivot_row += 1
        if pivot_row == rows:
            break

    return steps


def to_rref(matrix):
    A = np.array(matrix, dtype=float)
    rows, cols = A.shape
    row = 0
    for col in range(cols):
        if row >= rows:
            break
        pivot = None
        for r in range(row, rows):
            if A[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        A[row] /= A[row, col]
        for r in range(rows):
            if r != row:
                A[r] -= A[r, col] * A[row]
        row += 1
    return A

def inverse_matrix(matrix):
    A = np.array(matrix, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("正方行列のみ逆行列を計算できます")
    I = np.identity(n)
    AI = np.hstack((A, I))

    # Gauss-Jordan elimination
    for i in range(n):
        pivot = AI[i, i]
        if pivot == 0:
            for j in range(i+1, n):
                if AI[j, i] != 0:
                    AI[[i, j]] = AI[[j, i]]
                    pivot = AI[i, i]
                    break
            else:
                raise ValueError("逆行列が存在しません（特異行列）")
        AI[i] = AI[i] / pivot
        for j in range(n):
            if j != i:
                AI[j] = AI[j] - AI[j, i] * AI[i]
    return AI[:, n:]

# サンプル入力候補
sample_matrices = {
    "カスタム入力": None,
    "3x3 連立方程式系": [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
    "3x3 単位行列": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "2x2 正則": [[4, 7], [2, 6]],
    "2x2 特異（逆行列なし）": [[1, 2], [2, 4]],
}

# UI
st.title("行列計算アプリ")

# --- 入力サンプル選択
selected_sample = st.selectbox("サンプル行列を選ぶ（またはカスタム入力）", list(sample_matrices.keys()))

# --- 行列初期値
if selected_sample != "カスタム入力":
    default_matrix = sample_matrices[selected_sample]
    rows = len(default_matrix)
    cols = len(default_matrix[0])
    df_init = pd.DataFrame(default_matrix,
                           index=[f"行{i+1}" for i in range(rows)],
                           columns=[f"列{j+1}" for j in range(cols)])
else:
    rows = st.number_input("行数", min_value=1, max_value=10, value=3, step=1)
    cols = st.number_input("列数", min_value=1, max_value=10, value=3, step=1)
    df_init = pd.DataFrame(
        0,
        index=[f"行{i+1}" for i in range(rows)],
        columns=[f"列{j+1}" for j in range(cols)]
    )

# --- 編集可能な行列
st.markdown("### 行列を入力（セルをクリックして編集）")
edited_matrix = st.data_editor(df_init, num_rows="fixed", use_container_width=True)

matrix = edited_matrix.values.tolist()

# --- ボタン群
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("階段行列（REF）に変換"):
        try:
            ref = to_row_echelon(matrix)
            st.success("階段行列（REF）：")
            st.dataframe(ref)
        except Exception as e:
            st.error(f"エラー: {e}")
    if st.button("階段行列の変形過程を見る（学習モード）"):
        try:
            steps = to_row_echelon_with_steps(matrix)
            for i, (desc, mat) in enumerate(steps):
                st.markdown(f"### ステップ {i}: {desc}")
                st.dataframe(mat)
            st.success("変形完了！")
        except Exception as e:
            st.error(f"エラー: {e}")


with col2:
    if st.button("完全階段行列（RREF）に変換"):
        try:
            rref = to_rref(matrix)
            st.success("完全階段行列（RREF）：")
            st.dataframe(rref)
        except Exception as e:
            st.error(f"エラー: {e}")

with col3:
    if st.button("逆行列を求める"):
        try:
            inv = inverse_matrix(matrix)
            st.success("逆行列：")
            st.dataframe(inv)
        except Exception as e:
            st.error(f"エラー: {e}")
