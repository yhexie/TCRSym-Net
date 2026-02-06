import os
import glob
import argparse
import math

REF_DIR_DEFAULT = r"C:\Users\pc\Desktop\symmetric_exp\compare"
PRED_DIR_DEFAULT = r"C:\Users\pc\Desktop\symmetric_exp\compare\result"

OUTPUT_DIR = os.getcwd()


def _parse_floats_from_line(line: str):
    line = line.replace(",", " ").strip()
    if not line:
        return []
    return [float(x) for x in line.split()]


def read_sym_file(path: str):
    """Reads a .sym file and returns the point and the three normal vectors."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 4:
        raise ValueError(f"{os.path.basename(path)} 行数不足：至少需要4行")

    point = _parse_floats_from_line(lines[0])
    n1 = _parse_floats_from_line(lines[1])
    n2 = _parse_floats_from_line(lines[2])
    n3 = _parse_floats_from_line(lines[3])

    if len(point) != 3:
        raise ValueError(f"{os.path.basename(path)} 第1行无法解析为点坐标: {lines[0]}")
    if not (len(n1) == 3 and len(n2) == 3 and len(n3) == 3):
        raise ValueError(f"{os.path.basename(path)} 第2-4行必须每行3个数")

    return point, [n1, n2, n3]


def _norm(v):
    return math.sqrt(sum(c*c for c in v))


def _normalize(v):
    n = _norm(v)
    if n == 0:
        return None
    return [c / n for c in v]


def _dot(v1, v2):
    return sum(c1 * c2 for c1, c2 in zip(v1, v2))


def _angle_deg(n_ref, n_pred):
    a = _normalize(n_ref)
    b = _normalize(n_pred)
    if a is None or b is None:
        return None

    dot_product = abs(_dot(a, b))
    dot_product = max(-1.0, min(1.0, dot_product))
    return math.degrees(math.acos(dot_product))


def _point_to_plane_dist(point_pred, point_ref, normal_ref):
    """Calculates the distance from a predicted point to a reference plane."""
    n_ref_normalized = _normalize(normal_ref)
    if n_ref_normalized is None:
        return None
    
    vec_p_pred_to_p_ref = [pp - pr for pp, pr in zip(point_pred, point_ref)]
    
    distance = abs(_dot(n_ref_normalized, vec_p_pred_to_p_ref))
    return distance

def _best_match_angle(ref_n, pred_normals):
    best_idx, best_ang = -1, float('inf')
    for i, pn in enumerate(pred_normals):
        ang = _angle_deg(ref_n, pn)
        if ang is not None and ang < best_ang:
            best_ang = ang
            best_idx = i
    return (best_idx, best_ang) if best_idx != -1 else (None, None)


def _suffix_from_model_name(model_name: str) -> str:
    return {
        "pct": "_pct.txt",
        "mlp": "_mlp.txt",
        "pointnet": "_pointnet.txt",
    }.get(model_name.lower(), f"_{model_name}.txt")


def _ref_to_pred_suffix_for_match(model_name: str) -> str:
    return {
        "pct": ".eval_pct.sym",
        "mlp": ".eval_pointmlp.sym",
        "pointnet": ".eval.sym",
    }.get(model_name.lower(), ".eval.sym")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="pct", choices=["pct", "mlp", "pointnet"], help="模型名称，决定输入和输出文件后缀 (默认: pct)")
    parser.add_argument("--angle_thresh_deg", type=float, default=5.0, help="法向量匹配的角度阈值(度) (默认: 5.0)")
    parser.add_argument("--ref_dir", default=REF_DIR_DEFAULT, help="参考 .sym 文件目录")
    parser.add_argument("--pred_dir", default=PRED_DIR_DEFAULT, help="预测 .sym 文件目录")
    args = parser.parse_args()

    suffix = _suffix_from_model_name(args.model_name)
    match_pred_suffix = _ref_to_pred_suffix_for_match(args.model_name)

    out_report = os.path.join(OUTPUT_DIR, f"evaluation_report{suffix}")

    ref_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*.sym")))
    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, f"*{match_pred_suffix}")))

    pred_by_name = {os.path.basename(p): p for p in pred_paths}

    report_lines = []
    matched_ref_normals_count = 0
    total_ref_normals_count = 0
    all_best_angles = []
    matched_best_angles = []
    matched_distances = []

    for ref_path in ref_paths:
        ref_base = os.path.splitext(os.path.basename(ref_path))[0]
        target_pred_name = f"{ref_base}{match_pred_suffix}"

        if target_pred_name not in pred_by_name:
            continue

        pred_path = pred_by_name[target_pred_name]

        try:
            ref_point, ref_normals = read_sym_file(ref_path)
            pred_point, pred_normals = read_sym_file(pred_path)
        except Exception as e:
            print(f"[跳过] 文件解析失败: {e}")
            continue

        for i, ref_n in enumerate(ref_normals):
            total_ref_normals_count += 1
            best_pred_idx, best_ang = _best_match_angle(ref_n, pred_normals)

            dist = None
            is_match = 0
            if best_ang is not None and best_ang < args.angle_thresh_deg:
                is_match = 1
                matched_ref_normals_count += 1
                matched_best_angles.append(best_ang)
                dist = _point_to_plane_dist(pred_point, ref_point, ref_n)
                if dist is not None:
                    matched_distances.append(dist)
            
            if best_ang is not None:
                all_best_angles.append(best_ang)

            report_lines.append(
                f"{os.path.basename(ref_path)}\t{os.path.basename(pred_path)}\t{i+1}\t"
                f"{(best_pred_idx + 1) if best_pred_idx is not None else 'N/A'}\t"
                f"{best_ang:.6f if best_ang is not None else 'NaN'}\t"
                f"{dist:.6f if dist is not None else 'NaN'}\t{is_match}"
            )

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("ref_file\tpred_file\tref_normal_idx\tbest_pred_normal_idx\tbest_angle_deg\tpoint_plane_dist\tis_match\n")
        for line in report_lines:
            f.write(f"{line}\n")

        # 写入统计摘要
        f.write("\n--- 摘要 ---\n")
        f.write(f"模型名称: {args.model_name}\n")
        f.write(f"角度阈值 (度): {args.angle_thresh_deg}\n")
        f.write(f"总参考法向量数: {total_ref_normals_count}\n")
        f.write(f"匹配上的参考法向量数: {matched_ref_normals_count}\n")
        if total_ref_normals_count > 0:
            f.write(f"匹配率: {matched_ref_normals_count / total_ref_normals_count:.4f}\n")
        
        if all_best_angles:
            f.write(f"\n所有最佳角度 (度) - 平均: {sum(all_best_angles)/len(all_best_angles):.4f}, 中位数: {sorted(all_best_angles)[len(all_best_angles)//2]:.4f}\n")
        if matched_best_angles:
            f.write(f"匹配上的角度 (度) - 平均: {sum(matched_best_angles)/len(matched_best_angles):.4f}, 中位数: {sorted(matched_best_angles)[len(matched_best_angles)//2]:.4f}\n")
        if matched_distances:
            f.write(f"匹配上的距离 - 平均: {sum(matched_distances)/len(matched_distances):.4f}, 中位数: {sorted(matched_distances)[len(matched_distances)//2]:.4f}\n")

    print(f"评估报告已生成: {out_report}")

if __name__ == "__main__":
    main()
