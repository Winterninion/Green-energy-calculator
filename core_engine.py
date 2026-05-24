import re
from typing import Any, Optional

import numpy as np
import pandas as pd


class PVConsumptionCalculator:
    def __init__(
        self,
        pv_capacity_mw: float = 10.6,
        wind_capacity_mw: float = 0.0,
        wind_target_hours: float = 0.0,
        grid_capacity_mva: float = 15.0,
        fixed_load_mw: float = 0.0,
        storage_power_mw: float = 5.0,
        storage_energy_mwh: float = 10.264,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        min_soc_percent: float = 5.0,
        max_soc_percent: float = 100.0,
        initial_soc_percent: float = 5.0,
        enable_storage: bool = True,
    ):
        """
        光伏 + 风电 + 储能绿电直连消纳测算核心。

        兼容口径：
        - 负荷表：支持 15分钟/30分钟/小时功率曲线，自动折算为小时电量 kWh。
        - 光伏表：支持单位 MW 曲线、1~8760 小时序号、以及“15MW”等总出力曲线自动折算。
        - 风电表：支持完整日期时间、日期+时间、1~8760 小时序号；支持“5MW机型/5MW”等样机总出力曲线自动按风电规模放大。
        - 固定负荷：支持在合并后的负荷曲线上叠加一个固定 MW 负荷。
        - 储能：新能源先供负荷，富余电量充电，负荷缺口时放电，剩余富余为上网，剩余缺口为下网。
        """
        self.pv_capacity_mw = max(float(pv_capacity_mw or 0), 0.0)
        self.wind_capacity_mw = max(float(wind_capacity_mw or 0), 0.0)
        self.wind_target_hours = max(float(wind_target_hours or 0), 0.0)
        self.grid_capacity_mva = max(float(grid_capacity_mva or 0), 0.0)
        self.fixed_load_mw = max(float(fixed_load_mw or 0), 0.0)
        self.storage_power_mw = max(float(storage_power_mw or 0), 0.0)
        self.storage_energy_mwh = max(float(storage_energy_mwh or 0), 0.0)
        self.charge_efficiency = self._normalize_eff(charge_efficiency)
        self.discharge_efficiency = self._normalize_eff(discharge_efficiency)
        self.min_soc_percent = float(min_soc_percent)
        self.max_soc_percent = float(max_soc_percent)
        self.initial_soc_percent = float(initial_soc_percent)
        self.enable_storage = bool(enable_storage) and self.storage_power_mw > 0 and self.storage_energy_mwh > 0
        self.diagnostics: list[str] = []

        if self.min_soc_percent < 0 or self.max_soc_percent > 100 or self.min_soc_percent >= self.max_soc_percent:
            raise ValueError("储能 SOC 下限/上限设置不合法：请保证 0 <= 下限 < 上限 <= 100。")
        if not (self.min_soc_percent <= self.initial_soc_percent <= self.max_soc_percent):
            raise ValueError("储能初始 SOC 必须位于 SOC 下限和上限之间。")

    @staticmethod
    def _normalize_eff(value: float) -> float:
        value = float(value)
        if value > 1:
            value /= 100.0
        if value <= 0 or value > 1:
            raise ValueError("储能充/放电效率必须在 0~1 或 0~100% 范围内。")
        return value

    # ========================= 文件与表头兼容 =========================
    def _load_file_smart(self, file_obj: Any) -> pd.DataFrame:
        """兼容 csv / xls / xlsx；自动处理前几行说明文字、合并/重复表头。"""
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        
        # file_obj 可能是 FastAPI 的 BytesIO，也可能是本地路径字符串。
        import os
        source_filename = str(file_obj) if isinstance(file_obj, (str, bytes, bytearray)) else str(getattr(file_obj, "name", "") or "")
        filename = source_filename.lower()

        try:
            if filename.endswith(".xls"):
                try:
                    df = pd.read_excel(file_obj, engine="xlrd")
                except ImportError as e:
                    raise ImportError(
                        "检测到你上传的是 .xls 老版 Excel 文件。读取 .xls 需要安装 xlrd，"
                        "请在终端运行：py -m pip install xlrd；或者把 .xls 另存为 .xlsx 后再上传。"
                    ) from e
            elif filename.endswith((".xlsx", ".xlsm")):
                df = pd.read_excel(file_obj, engine="openpyxl")
            else:
                # 先按 Excel 试，再按 CSV 兜底
                df = pd.read_excel(file_obj)
        except ImportError:
            raise
        except Exception:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            try:
                df = pd.read_csv(file_obj, encoding="utf-8-sig")
            except UnicodeDecodeError:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding="gbk")

        if self._looks_like_bad_header(df) and hasattr(file_obj, "seek"):
            try:
                file_obj.seek(0)
                if filename.endswith(".xls"):
                    raw = pd.read_excel(file_obj, header=None, nrows=50, engine="xlrd")
                elif filename.endswith((".xlsx", ".xlsm")):
                    raw = pd.read_excel(file_obj, header=None, nrows=50, engine="openpyxl")
                else:
                    raw = pd.read_excel(file_obj, header=None, nrows=50)
                header_row = self._guess_header_row(raw)
                if header_row is not None:
                    file_obj.seek(0)
                    if filename.endswith(".xls"):
                        df = pd.read_excel(file_obj, header=header_row, engine="xlrd")
                    elif filename.endswith((".xlsx", ".xlsm")):
                        df = pd.read_excel(file_obj, header=header_row, engine="openpyxl")
                    else:
                        df = pd.read_excel(file_obj, header=header_row)
            except ImportError:
                raise
            except Exception:
                pass

        df = self._repair_merged_or_repeated_header(df)
        df = df.dropna(how="all").copy()
        
        df.attrs["source_name"] = source_filename
        return df

    @staticmethod
    def _looks_like_bad_header(df: pd.DataFrame) -> bool:
        cols = [str(c) for c in df.columns]
        return len(cols) > 0 and sum(c.startswith("Unnamed") for c in cols) >= max(1, len(cols) // 3)

    @staticmethod
    def _guess_header_row(raw: pd.DataFrame) -> Optional[int]:
        keywords = ("时间", "日期", "功率", "负荷", "电量", "发电", "出力", "曲线", "kw", "kwh", "mw")
        best_row, best_score = None, 0
        for idx, row in raw.iterrows():
            cells = [str(x).strip().lower() for x in row.tolist() if pd.notna(x)]
            score = sum(any(k.lower() in cell for k in keywords) for cell in cells)
            if score > best_score:
                best_row, best_score = int(idx), score
        return best_row if best_score >= 2 else None

    @staticmethod
    def _is_probably_header_cell(x: Any) -> bool:
        if pd.isna(x):
            return False
        s = str(x).strip()
        if not s:
            return False
        keywords = ("时间", "日期", "功率", "负荷", "电量", "发电", "出力", "曲线", "kw", "kwh", "mw", "总有功")
        if any(k.lower() in s.lower() for k in keywords):
            return True
        try:
            float(s)
            return False
        except Exception:
            return False

    def _repair_merged_or_repeated_header(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理类似列名为 数据日期/时间/Unnamed: 2，首行才写着 总有功功率 的表。"""
        if df.empty:
            return df
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        best_i, best_score = None, 0
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            score = 0
            for col, val in zip(df.columns, row.tolist()):
                col_s = str(col).strip()
                if self._is_probably_header_cell(val):
                    if col_s.startswith("Unnamed") or str(val).strip() == col_s or self._is_probably_header_cell(col_s):
                        score += 1
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= 2:
            row = df.iloc[best_i]
            new_cols, changed = [], False
            for col, val in zip(df.columns, row.tolist()):
                col_s = str(col).strip()
                if self._is_probably_header_cell(val) and (col_s.startswith("Unnamed") or str(val).strip() == col_s):
                    new_cols.append(str(val).strip())
                    changed = True
                else:
                    new_cols.append(col_s)
            if changed:
                df = df.iloc[best_i + 1:].copy()
                df.columns = self._dedupe_columns(new_cols)
        return df

    @staticmethod
    def _dedupe_columns(cols: list[str]) -> list[str]:
        seen, out = {}, []
        for c in cols:
            c = str(c).strip() or "Unnamed"
            if c not in seen:
                seen[c] = 0
                out.append(c)
            else:
                seen[c] += 1
                out.append(f"{c}_{seen[c]}")
        return out

    # ========================= 时间与数值列识别 =========================
    @staticmethod
    def _is_sequential_hour_index(series: pd.Series) -> bool:
        nums = pd.to_numeric(series, errors="coerce").dropna()
        if len(nums) < 24 or len(nums) / max(len(series), 1) < 0.9:
            return False
        diffs = nums.diff().dropna()
        if diffs.empty:
            return False
        one_step_ratio = float((diffs.round(6) == 1).mean())
        min_v, max_v = float(nums.min()), float(nums.max())
        return one_step_ratio > 0.95 and min_v in (0.0, 1.0) and 1000 <= max_v <= 9000

    @staticmethod
    def _virtual_hour_datetime(n: int) -> pd.Series:
        return pd.Series(pd.date_range("2001-01-01 00:00", periods=n, freq="h"))

    @staticmethod
    def _extract_capacity_from_text(text: Any) -> Optional[float]:
        """从“15MW”“5MW机型”“20台5MW”等文本中提取容量 MW。"""
        name = str(text or "").strip()
        patterns = [r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:mw|MW|兆瓦|万千瓦)"]
        matches = []
        for pat in patterns:
            for m in re.finditer(pat, name):
                cap = float(m.group(1))
                if "万千瓦" in m.group(0):
                    cap *= 10.0
                if cap > 0:
                    matches.append(cap)
        if not matches:
            return None
        # 文件名里可能有日期“260416”，但只有带 MW 的数字才会命中；若有多个 MW，优先较小样机容量。
        return min(matches)

    @staticmethod
    def _find_col(df: pd.DataFrame, include: list[str], exclude: Optional[list[str]] = None) -> Optional[str]:
        exclude = exclude or []
        for col in df.columns:
            name = str(col).strip().lower()
            if any(k.lower() in name for k in include) and not any(k.lower() in name for k in exclude):
                return col
        return None

    @staticmethod
    def _parse_date_and_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
        date_text = date_series.astype(str).str.strip()
        time_text = time_series.astype(str).str.strip()
        time_text = time_text.str.replace(r"^0 days\s+", "", regex=True).str.replace(r"\.0$", "", regex=True)
        combined = date_text + " " + time_text
        dt = pd.to_datetime(combined, errors="coerce")
        bad = dt.isna()
        if bad.any():
            dt2 = pd.to_datetime(combined[bad], errors="coerce", dayfirst=True)
            dt.loc[bad] = dt2
        return dt

    @staticmethod
    def _infer_interval_hours(dt: pd.Series) -> float:
        s = pd.to_datetime(dt, errors="coerce").dropna().sort_values()
        if len(s) < 2:
            return 1.0
        diffs = s.diff().dropna().dt.total_seconds() / 3600.0
        diffs = diffs[(diffs > 0) & (diffs <= 24)]
        if diffs.empty:
            return 1.0
        med = float(diffs.median())
        for candidate in [1 / 60, 5 / 60, 10 / 60, 0.25, 0.5, 1.0, 2.0]:
            if abs(med - candidate) < 1e-4:
                return float(candidate)
        return med if med > 0 else 1.0

    @staticmethod
    def _find_value_col(df: pd.DataFrame, data_type_name: str, exclude_cols: set[str]) -> str:
        cols = [c for c in df.columns if c not in exclude_cols]
        if data_type_name == "用电":
            positive_keywords = ["总有功功率", "有功功率", "功率", "负荷", "用电", "电量", "kw", "kwh"]
        elif data_type_name == "风电":
            positive_keywords = ["风电", "风机", "出力", "发电", "功率", "新能源", "曲线", "kw", "mw"]
        else:
            positive_keywords = ["光伏发电功率", "发电功率", "发电", "光伏", "新能源", "出力", "曲线", "功率", "kw", "mw"]
        negative_keywords = ["资产", "编号", "用户", "名称", "户号", "id", "code", "no"]

        scored = []
        for col in cols:
            name = str(col).strip().lower()
            if any(k.lower() in name for k in negative_keywords):
                continue
            numeric = pd.to_numeric(df[col], errors="coerce")
            numeric_count = int(numeric.notna().sum())
            if numeric_count == 0:
                continue
            keyword_score = sum(k.lower() in name for k in positive_keywords)
            nonzero_score = int((numeric.fillna(0) != 0).sum())
            scored.append((keyword_score, nonzero_score, numeric_count, col))
        if scored:
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            return scored[0][3]
        raise ValueError(f"{data_type_name}表找不到可用的数值列。当前表头: {list(df.columns)}")

    def _auto_extract_data(self, df: pd.DataFrame, data_type_name: str) -> pd.DataFrame:
        """自动识别日期列、时间列、数值列，并统一输出每个原始步长的 energy_kwh。"""
        df = df.dropna(how="all").copy()
        df.columns = [str(c).strip() for c in df.columns]
        source_name = str(df.attrs.get("source_name", ""))

        date_col = self._find_col(df, include=["日期", "date"], exclude=["时间", "time"])
        data_date_col = self._find_col(df, include=["数据日期"])
        if data_date_col:
            date_col = data_date_col
        time_col = self._find_col(df, include=["时间", "time", "时刻"])
        datetime_col = self._find_col(df, include=["datetime", "日期时间", "时间戳"])

        used_virtual_hour_index = False
        if datetime_col:
            dt = pd.to_datetime(df[datetime_col], errors="coerce")
        elif date_col and time_col and date_col != time_col:
            dt = self._parse_date_and_time(df[date_col], df[time_col])
        elif date_col:
            if self._is_sequential_hour_index(df[date_col]):
                dt = self._virtual_hour_datetime(len(df))
                used_virtual_hour_index = True
                self.diagnostics.append(f"{data_type_name}表“{date_col}”列识别为 1~8760 小时序号，已自动生成时间轴。")
            else:
                dt = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        elif time_col:
            if self._is_sequential_hour_index(df[time_col]):
                dt = self._virtual_hour_datetime(len(df))
                used_virtual_hour_index = True
                self.diagnostics.append(f"{data_type_name}表“{time_col}”列识别为 1~8760 小时序号，已自动生成时间轴。")
            else:
                dt = pd.to_datetime(df[time_col], errors="coerce")
                if dt.notna().sum() == 0:
                    raise ValueError(
                        f"{data_type_name}表只找到了时间列“{time_col}”，但无法解析为完整日期时间。"
                        f"请提供“日期/数据日期 + 时间”，或完整日期时间列。当前表头: {list(df.columns)}"
                    )
        else:
            raise ValueError(
                f"{data_type_name}表找不到日期/时间列。请确保存在“日期/数据日期”和“时间”，"
                f"或一个完整日期时间列。当前表头: {list(df.columns)}"
            )

        val_col = self._find_value_col(df, data_type_name, exclude_cols={c for c in [date_col, time_col, datetime_col] if c})
        raw_value = pd.to_numeric(df[val_col], errors="coerce")
        out = pd.DataFrame({"datetime": dt, "standard_value": raw_value}).dropna(subset=["datetime"])
        out["standard_value"] = out["standard_value"].fillna(0).astype(float)
        out = out.sort_values("datetime").reset_index(drop=True)
        interval_hours = 1.0 if used_virtual_hour_index else self._infer_interval_hours(out["datetime"])
        out["interval_hours"] = interval_hours
        out["energy_kwh"] = out["standard_value"] * interval_hours

        # 光伏、风电根据列名/文件名中的容量自动折算。默认：负荷/光伏单位曲线为 kW 或 kW/MW。
        if data_type_name == "光伏":
            header_capacity_mw = self._extract_capacity_from_text(val_col)
            if header_capacity_mw is not None and header_capacity_mw > 0:
                # 例如“15MW”列，数值是 MW 总出力：MW * 1000 / 15 = kW/MW
                out["energy_kwh"] = out["standard_value"] * 1000.0 * interval_hours / header_capacity_mw
                self.diagnostics.append(
                    f"光伏数值列“{val_col}”识别为 {header_capacity_mw:g}MW 光伏总出力，已折算为单位 MW 曲线后按输入容量放大。"
                )

        if data_type_name == "风电":
            source_capacity_mw = self._extract_capacity_from_text(val_col) or self._extract_capacity_from_text(source_name)
            col_name = str(val_col).lower()
            if source_capacity_mw is not None and source_capacity_mw > 0:
                # 常见样例：“5MW机型8760h出力曲线.xlsx”，列为“出力(kW)”，表示 5MW 机组总出力 kW。
                if "mw" in col_name and "kw" not in col_name:
                    base_energy = out["standard_value"] * 1000.0 * interval_hours
                else:
                    base_energy = out["standard_value"] * interval_hours
                out["energy_kwh"] = base_energy / source_capacity_mw
                self.diagnostics.append(
                    f"风电曲线识别为 {source_capacity_mw:g}MW 样机/场站总出力，已折算为单位 MW 曲线后按输入风电规模放大。"
                )
            else:
                # 没找到样机容量时，默认该列已经是单位 MW 风电曲线 kW/MW。
                self.diagnostics.append("风电曲线未从列名/文件名识别出样机容量，默认按单位 MW 曲线处理。")

        out.attrs["value_col"] = str(val_col)
        out.attrs["used_virtual_hour_index"] = used_virtual_hour_index
        out["hour_datetime"] = out["datetime"].dt.floor("h")
        out["month"] = out["hour_datetime"].dt.month.astype(int)
        out["day"] = out["hour_datetime"].dt.day.astype(int)
        out["hour"] = out["hour_datetime"].dt.hour.astype(int)
        return out[["datetime", "hour_datetime", "month", "day", "hour", "standard_value", "energy_kwh", "interval_hours"]]

    # ========================= 三类数据处理 =========================
    def _process_single_load_file(self, load_file_path: Any, source_index: int = 1) -> pd.DataFrame:
        """读取单个负荷文件并折算为小时电量。

        输出中的 load_kw 实际是该小时电量 kWh。保留 source_name，便于多个主变/线路
        负荷曲线按同一小时累加，并统计该小时有多少条负荷曲线参与。
        """
        df_load_raw = self._load_file_smart(load_file_path)
        source_name = str(df_load_raw.attrs.get("source_name", "") or f"load_{source_index}")
        df_load = self._auto_extract_data(df_load_raw, "用电")
        hourly = df_load.groupby(["hour_datetime", "month", "day", "hour"], as_index=False)["energy_kwh"].sum()
        hourly = hourly.rename(columns={"hour_datetime": "datetime", "energy_kwh": "load_kw"})
        hourly["load_source_name"] = source_name
        hourly["load_source_index"] = int(source_index)
        hourly["source_has_data"] = 1
        hourly["source_is_active"] = (hourly["load_kw"].abs() > 1e-9).astype(int)
        return hourly

    def process_load_data(self, load_file_path: Any) -> pd.DataFrame:
        """处理一个或多个负荷文件。

        支持用户一次上传任意数量的主变/副变/关口表：
        - 同一小时存在多张表时自动累加为企业总负荷；
        - 某些时段只有一张旧总表、后续时段拆成多张分表时，会按真实时间自动衔接；
        - 返回 load_curve_count / active_load_curve_count，便于前端提示不同时间参与计算的负荷曲线数量。
        """
        if isinstance(load_file_path, (list, tuple)):
            load_files = list(load_file_path)
        else:
            load_files = [load_file_path]
        load_files = [f for f in load_files if f is not None]
        if not load_files:
            raise ValueError("请至少上传 1 个用电负荷表。")

        pieces = []
        for i, f in enumerate(load_files, start=1):
            one = self._process_single_load_file(f, i)
            if one.empty:
                self.diagnostics.append(f"第 {i} 个负荷表解析后为空，已忽略。")
                continue
            pieces.append(one)
            self.diagnostics.append(
                f"负荷表 {i} 已解析：{one['load_source_name'].iloc[0]}，"
                f"小时点数 {len(one)}，电量 {one['load_kw'].sum()/10000:.2f} 万kWh。"
            )

        if not pieces:
            raise ValueError("所有负荷表解析后均为空，请检查上传文件。")

        all_load = pd.concat(pieces, ignore_index=True)
        grouped = all_load.groupby(["datetime", "month", "day", "hour"], as_index=False).agg(
            load_kw=("load_kw", "sum"),
            load_curve_count=("load_source_name", "nunique"),
            active_load_curve_count=("source_is_active", "sum"),
        )
        grouped = grouped.sort_values("datetime").reset_index(drop=True)

        if self.fixed_load_mw > 0 and not grouped.empty:
            fixed_load_kwh_per_hour = self.fixed_load_mw * 1000.0
            grouped["uploaded_load_kw"] = grouped["load_kw"].astype(float)
            grouped["fixed_load_kw"] = fixed_load_kwh_per_hour
            grouped["load_kw"] = grouped["load_kw"].astype(float) + fixed_load_kwh_per_hour
            self.diagnostics.append(
                f"已叠加固定负荷 {self.fixed_load_mw:g} MW：按每个小时增加 {fixed_load_kwh_per_hour:.2f} kWh 计入总负荷。"
            )
        else:
            grouped["uploaded_load_kw"] = grouped["load_kw"].astype(float)
            grouped["fixed_load_kw"] = 0.0

        # 如果多个负荷文件存在重叠小时，默认按分表累加。若用户误把“总表”和“分表”
        # 覆盖在同一时段上传，诊断信息会提醒其检查。
        overlap_hours = int((grouped["load_curve_count"] > 1).sum())
        max_count = int(grouped["load_curve_count"].max()) if len(grouped) else 0
        self.diagnostics.append(
            f"已合并 {len(pieces)} 个负荷表：最大同时参与负荷曲线数 {max_count}，"
            f"多曲线叠加小时数 {overlap_hours}。"
        )
        return grouped

    def process_pv_data(self, pv_file_path: Any) -> pd.DataFrame:
        if pv_file_path is None or self.pv_capacity_mw <= 0:
            return pd.DataFrame(columns=["datetime", "month", "day", "hour", "pv_total_kw"])
        df_pv = self._load_file_smart(pv_file_path)
        df_pv = self._auto_extract_data(df_pv, "光伏")
        if (df_pv["interval_hours"].median() >= 0.99) and df_pv["hour_datetime"].duplicated().any():
            virtual_time = pd.date_range("2001-01-01 00:00", periods=len(df_pv), freq="h")
            df_pv = df_pv.copy()
            df_pv["hour_datetime"] = virtual_time
            df_pv["month"] = virtual_time.month.astype(int)
            df_pv["day"] = virtual_time.day.astype(int)
            df_pv["hour"] = virtual_time.hour.astype(int)
            self.diagnostics.append("光伏曲线存在重复小时，已按全年第 N 小时顺序重建时间轴。")
        hourly = df_pv.groupby(["hour_datetime", "month", "day", "hour"], as_index=False)["energy_kwh"].sum()
        hourly = hourly.rename(columns={"hour_datetime": "datetime", "energy_kwh": "pv_unit_kwh"})
        hourly["pv_total_kw"] = hourly["pv_unit_kwh"] * self.pv_capacity_mw
        return hourly[["datetime", "month", "day", "hour", "pv_total_kw"]]

    def process_wind_data(self, wind_file_path: Any) -> pd.DataFrame:
        if wind_file_path is None or self.wind_capacity_mw <= 0:
            return pd.DataFrame(columns=["datetime", "month", "day", "hour", "wind_total_kw"])
        df_wind = self._load_file_smart(wind_file_path)
        df_wind = self._auto_extract_data(df_wind, "风电")
        if (df_wind["interval_hours"].median() >= 0.99) and df_wind["hour_datetime"].duplicated().any():
            virtual_time = pd.date_range("2001-01-01 00:00", periods=len(df_wind), freq="h")
            df_wind = df_wind.copy()
            df_wind["hour_datetime"] = virtual_time
            df_wind["month"] = virtual_time.month.astype(int)
            df_wind["day"] = virtual_time.day.astype(int)
            df_wind["hour"] = virtual_time.hour.astype(int)
            self.diagnostics.append("风电曲线存在重复小时，已按全年第 N 小时顺序重建时间轴。")
        hourly = df_wind.groupby(["hour_datetime", "month", "day", "hour"], as_index=False)["energy_kwh"].sum()
        hourly = hourly.rename(columns={"hour_datetime": "datetime", "energy_kwh": "wind_unit_kwh"})
        hourly["wind_total_kw"] = hourly["wind_unit_kwh"] * self.wind_capacity_mw

        # 可选：按申报/设计口径的“等效利用小时数”对风电年发电量做校准。
        # 例如 5MW 样机曲线原始合计为 10012.068MWh，折算 100MW 后为 20024.14万kWh；
        # 若申报口径要求 100MW × 2000h = 20000万kWh，可在前端填 2000 自动缩放。
        if self.wind_target_hours > 0 and self.wind_capacity_mw > 0:
            target_total_kwh = self.wind_capacity_mw * self.wind_target_hours * 1000.0
            current_total_kwh = float(hourly["wind_total_kw"].sum())
            if current_total_kwh > 0:
                factor = target_total_kwh / current_total_kwh
                hourly["wind_total_kw"] = hourly["wind_total_kw"] * factor
                hourly["wind_unit_kwh"] = hourly["wind_unit_kwh"] * factor
                self.diagnostics.append(
                    f"风电年发电量已按等效利用小时数 {self.wind_target_hours:g}h 校准："
                    f"原始 {current_total_kwh/10000:.2f} 万kWh，校准后 {target_total_kwh/10000:.2f} 万kWh，缩放系数 {factor:.6f}。"
                )
        return hourly[["datetime", "month", "day", "hour", "wind_total_kw"]]

    # ========================= 合并与储能模拟 =========================
    def _align_generation_and_load(self, df_load: pd.DataFrame, df_pv: pd.DataFrame, df_wind: pd.DataFrame) -> pd.DataFrame:
        # 以“全年第 N 小时”为主轴对齐。
        # 旧版以负荷小时数为主轴，若负荷表只有 8736 点、风电典型年有 8760 点，
        # 会把最后 24 小时风电裁掉，导致新能源年发电量偏小。这里改为取三类曲线的
        # 最大长度，负荷缺失小时补 0；这样年发电量保持完整，消纳模拟中缺失负荷小时
        # 会自然表现为上网电量。
        df_load = df_load.sort_values("datetime").reset_index(drop=True).copy()
        df_pv = df_pv.sort_values("datetime").reset_index(drop=True).copy() if df_pv is not None and not df_pv.empty else df_pv
        df_wind = df_wind.sort_values("datetime").reset_index(drop=True).copy() if df_wind is not None and not df_wind.empty else df_wind

        lengths = [len(df_load)]
        if df_pv is not None and not df_pv.empty:
            lengths.append(len(df_pv))
        if df_wind is not None and not df_wind.empty:
            lengths.append(len(df_wind))
        n = max(lengths) if lengths else 0

        virtual_time = pd.date_range("2001-01-01 00:00", periods=n, freq="h")
        base = pd.DataFrame({
            "datetime": virtual_time,
            "month": virtual_time.month.astype(int),
            "day": virtual_time.day.astype(int),
            "hour": virtual_time.hour.astype(int),
            "load_kw": 0.0,
            "load_curve_count": 0.0,
            "active_load_curve_count": 0.0,
            "uploaded_load_kw": 0.0,
            "fixed_load_kw": 0.0,
            "pv_total_kw": 0.0,
            "wind_total_kw": 0.0,
        })

        def copy_by_order(target: pd.DataFrame, source: pd.DataFrame, source_col: str, target_col: str | None = None) -> pd.DataFrame:
            target_col = target_col or source_col
            if source is None or source.empty or source_col not in source.columns:
                return target
            source = source.sort_values("datetime").reset_index(drop=True)
            m = min(len(target), len(source))
            if m > 0:
                target.loc[:m - 1, target_col] = source[source_col].iloc[:m].to_numpy(dtype=float)
            return target

        for col in ["load_kw", "load_curve_count", "active_load_curve_count", "uploaded_load_kw", "fixed_load_kw"]:
            fallback = df_load["load_kw"] if col == "uploaded_load_kw" and col not in df_load.columns else pd.Series([0.0] * len(df_load))
            source = df_load.copy()
            if col not in source.columns:
                source[col] = fallback.to_numpy(dtype=float) if len(fallback) == len(source) else 0.0
            base = copy_by_order(base, source, col)

        base = copy_by_order(base, df_pv, "pv_total_kw")
        base = copy_by_order(base, df_wind, "wind_total_kw")

        if len(df_load) < n:
            self.diagnostics.append(
                f"负荷曲线小时数为 {len(df_load)}，新能源曲线最长为 {n}，已补齐 {n - len(df_load)} 个无负荷小时，避免裁剪全年风/光发电量。"
            )
        if df_pv is not None and not df_pv.empty and len(df_pv) < n:
            self.diagnostics.append(f"光伏曲线小时数为 {len(df_pv)}，少于计算主轴 {n}，缺失小时按 0 发电处理。")
        if df_wind is not None and not df_wind.empty and len(df_wind) < n:
            self.diagnostics.append(f"风电曲线小时数为 {len(df_wind)}，少于计算主轴 {n}，缺失小时按 0 发电处理。")

        base["renewable_total_kw"] = base["pv_total_kw"] + base["wind_total_kw"]
        return base

    def calculate(self, load_file_path: Any, pv_file_path: Any = None, wind_file_path: Any = None):
        df_load = self.process_load_data(load_file_path)
        df_pv = self.process_pv_data(pv_file_path)
        df_wind = self.process_wind_data(wind_file_path)
        df_merged = self._align_generation_and_load(df_load, df_pv, df_wind)
        if df_merged.empty:
            raise ValueError("负荷表解析后为空，请检查上传文件。")
        df_detail = self._simulate_storage(df_merged)
        # 申报约束：年上网电量比例不得超过新能源年发电量的 20%。
        # 超过部分不再计为上网电量，而转入年弃电量。
        df_detail = self._apply_annual_export_cap(df_detail, max_export_ratio=0.20)
        monthly_stats = self._build_monthly_stats(df_detail)
        summary = self._build_summary(df_detail)
        return summary, monthly_stats, df_detail

    def _simulate_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cap_kwh = self.storage_energy_mwh * 1000.0
        power_kwh_per_hour = self.storage_power_mw * 1000.0
        min_soc_kwh = cap_kwh * self.min_soc_percent / 100.0
        max_soc_kwh = cap_kwh * self.max_soc_percent / 100.0
        soc = cap_kwh * self.initial_soc_percent / 100.0

        records = []
        for _, row in out.iterrows():
            gen = max(float(row["renewable_total_kw"]), 0.0)
            load = max(float(row["load_kw"]), 0.0)
            direct_consumed = min(gen, load)
            surplus = max(gen - load, 0.0)
            deficit = max(load - gen, 0.0)

            charge_from_re = 0.0
            discharge_to_load = 0.0
            grid_export = surplus
            grid_purchase = deficit
            loss_kwh = 0.0

            if self.enable_storage:
                headroom = max(max_soc_kwh - soc, 0.0)
                max_charge_by_capacity = headroom / self.charge_efficiency if self.charge_efficiency > 0 else 0.0
                charge_from_re = min(surplus, power_kwh_per_hour, max_charge_by_capacity)
                soc += charge_from_re * self.charge_efficiency
                grid_export = surplus - charge_from_re
                loss_kwh += charge_from_re * (1.0 - self.charge_efficiency)

                available = max(soc - min_soc_kwh, 0.0)
                max_discharge_by_capacity = available * self.discharge_efficiency
                discharge_to_load = min(deficit, power_kwh_per_hour, max_discharge_by_capacity)
                soc -= discharge_to_load / self.discharge_efficiency if self.discharge_efficiency > 0 else 0.0
                grid_purchase = deficit - discharge_to_load
                loss_kwh += discharge_to_load * (1.0 / self.discharge_efficiency - 1.0)

            consumed_by_load = direct_consumed + discharge_to_load
            local_consumed_including_loss = direct_consumed + charge_from_re
            soc = min(max(soc, min_soc_kwh), max_soc_kwh) if cap_kwh > 0 else 0.0

            records.append({
                "direct_consumed_kw": direct_consumed,
                "storage_charge_kw": charge_from_re,
                "storage_discharge_kw": discharge_to_load,
                "storage_loss_kw": loss_kwh,
                "grid_export_kw": max(grid_export, 0.0),
                "grid_purchase_kw": max(grid_purchase, 0.0),
                "curtailment_kw": 0.0,
                "consumed_kw": consumed_by_load,
                "local_consumed_including_loss_kw": local_consumed_including_loss,
                "storage_soc_kwh": soc,
                "storage_soc_percent": (soc / cap_kwh * 100.0) if cap_kwh > 0 else 0.0,
            })
        return pd.concat([out, pd.DataFrame(records)], axis=1)

    def _apply_annual_export_cap(self, df: pd.DataFrame, max_export_ratio: float = 0.20) -> pd.DataFrame:
        """限制年上网电量比例。

        口径：年上网电量 <= 新能源年发电量 × max_export_ratio。
        若储能模拟后的年上网电量超过该上限，则把超过部分从 grid_export_kw
        转入 curtailment_kw。为了让月度图表和明细表仍然自洽，超过部分按各小时
        原始上网电量占比分摊扣减。
        """
        out = df.copy()
        total_generation = float(out.get("renewable_total_kw", pd.Series(dtype=float)).sum())
        total_export = float(out.get("grid_export_kw", pd.Series(dtype=float)).sum())

        if total_generation <= 0 or total_export <= 0:
            return out

        max_export_ratio = max(float(max_export_ratio), 0.0)
        allowed_export = total_generation * max_export_ratio
        if total_export <= allowed_export + 1e-9:
            self.diagnostics.append(
                f"年上网电量比例约束：上限 {max_export_ratio * 100:.0f}%，当前上网 {total_export/10000:.2f} 万kWh，未触发弃电调整。"
            )
            return out

        excess_export = total_export - allowed_export
        export_series = out["grid_export_kw"].astype(float).clip(lower=0.0)
        if export_series.sum() <= 0:
            return out

        # 按各小时原始上网电量比例分摊“超额上网转弃电”，避免只截断年末导致月度图表失真。
        reduction = export_series / export_series.sum() * excess_export
        reduction = np.minimum(reduction, export_series)
        out["grid_export_kw"] = export_series - reduction
        out["curtailment_kw"] = out.get("curtailment_kw", 0.0) + reduction

        actual_export = float(out["grid_export_kw"].sum())
        actual_curtailment_added = float(reduction.sum())
        self.diagnostics.append(
            f"年上网电量比例超过 {max_export_ratio * 100:.0f}%：原上网 {total_export/10000:.2f} 万kWh，"
            f"允许上网 {allowed_export/10000:.2f} 万kWh，已将超额 {actual_curtailment_added/10000:.2f} 万kWh 转为弃电。"
        )
        # 极小浮点误差不会影响展示；这里保留诊断便于核验。
        if abs(actual_export - allowed_export) > 1e-4:
            self.diagnostics.append(
                f"年上网约束校核：调整后上网 {actual_export/10000:.2f} 万kWh，占新能源发电量 {self._safe_div(actual_export, total_generation)*100:.2f}%。"
            )
        return out

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if float(denominator) > 0 else 0.0

    def _build_monthly_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "pv_total_kw", "wind_total_kw", "renewable_total_kw", "load_kw",
            "uploaded_load_kw", "fixed_load_kw",
            "direct_consumed_kw", "storage_charge_kw", "storage_discharge_kw",
            "storage_loss_kw", "consumed_kw", "local_consumed_including_loss_kw",
            "grid_export_kw", "grid_purchase_kw", "curtailment_kw",
        ]
        monthly = df.groupby("month", as_index=True)[cols].sum()
        if "load_curve_count" in df.columns:
            monthly["load_curve_count_avg"] = df.groupby("month")["load_curve_count"].mean()
            monthly["load_curve_count_max"] = df.groupby("month")["load_curve_count"].max()
        else:
            monthly["load_curve_count_avg"] = 1.0
            monthly["load_curve_count_max"] = 1.0
        if "active_load_curve_count" in df.columns:
            monthly["active_load_curve_count_avg"] = df.groupby("month")["active_load_curve_count"].mean()
            monthly["active_load_curve_count_max"] = df.groupby("month")["active_load_curve_count"].max()
        else:
            monthly["active_load_curve_count_avg"] = 1.0
            monthly["active_load_curve_count_max"] = 1.0
        monthly["月度直接消纳率(%)"] = np.where(monthly["renewable_total_kw"] > 0, monthly["direct_consumed_kw"] / monthly["renewable_total_kw"] * 100, 0).round(2)
        monthly["月度负荷侧绿电消纳率(%)"] = np.where(monthly["renewable_total_kw"] > 0, monthly["consumed_kw"] / monthly["renewable_total_kw"] * 100, 0).round(2)
        monthly["月度园区消纳率_含储能损耗(%)"] = np.where(monthly["renewable_total_kw"] > 0, monthly["local_consumed_including_loss_kw"] / monthly["renewable_total_kw"] * 100, 0).round(2)
        monthly["月度上网率(%)"] = np.where(monthly["renewable_total_kw"] > 0, monthly["grid_export_kw"] / monthly["renewable_total_kw"] * 100, 0).round(2)
        monthly["月度绿电占用电比例(%)"] = np.where(monthly["load_kw"] > 0, monthly["consumed_kw"] / monthly["load_kw"] * 100, 0).round(2)
        return monthly

    def _build_summary(self, df: pd.DataFrame) -> dict[str, float]:
        total_pv = float(df["pv_total_kw"].sum())
        total_wind = float(df["wind_total_kw"].sum())
        total_generation = float(df["renewable_total_kw"].sum())
        total_load = float(df["load_kw"].sum())
        total_uploaded_load = float(df.get("uploaded_load_kw", df["load_kw"]).sum())
        total_fixed_load = float(df.get("fixed_load_kw", pd.Series([0.0] * len(df))).sum())
        total_direct = float(df["direct_consumed_kw"].sum())
        total_charge = float(df["storage_charge_kw"].sum())
        total_discharge = float(df["storage_discharge_kw"].sum())
        total_loss = float(df["storage_loss_kw"].sum())
        total_consumed = float(df["consumed_kw"].sum())
        total_local_including_loss = float(df["local_consumed_including_loss_kw"].sum())
        total_export = float(df["grid_export_kw"].sum())
        total_purchase = float(df["grid_purchase_kw"].sum())
        total_curtailment = float(df["curtailment_kw"].sum())
        avg_load_rate = self._safe_div(total_load, self.grid_capacity_mva * 1000.0 * len(df)) if self.grid_capacity_mva > 0 else 0.0
        renewable_utilization = self._safe_div(total_generation - total_curtailment, total_generation)
        max_load_curve_count = float(df.get("load_curve_count", pd.Series([1])).max()) if len(df) else 0.0
        avg_load_curve_count = float(df.get("load_curve_count", pd.Series([1])).mean()) if len(df) else 0.0
        max_active_load_curve_count = float(df.get("active_load_curve_count", pd.Series([1])).max()) if len(df) else 0.0

        return {
            "total_pv_kwh": total_pv,
            "total_wind_kwh": total_wind,
            "total_generation_kwh": total_generation,
            "total_load_kwh": total_load,
            "net_load_after_internal_storage_kwh": total_purchase,
            "average_load_rate": avg_load_rate,
            "wind_capacity_mw": self.wind_capacity_mw,
            "wind_target_hours": self.wind_target_hours,
            "wind_equivalent_hours_actual": self._safe_div(total_wind, self.wind_capacity_mw * 1000.0) if self.wind_capacity_mw > 0 else 0.0,
            "max_load_curve_count": max_load_curve_count,
            "avg_load_curve_count": avg_load_curve_count,
            "max_active_load_curve_count": max_active_load_curve_count,
            "total_direct_consumed_kwh": total_direct,
            "total_storage_charge_kwh": total_charge,
            "total_storage_discharge_kwh": total_discharge,
            "total_storage_loss_kwh": total_loss,
            "total_consumed_kwh": total_consumed,
            "total_local_consumed_including_loss_kwh": total_local_including_loss,
            "total_grid_export_kwh": total_export,
            "total_grid_purchase_kwh": total_purchase,
            "total_curtailment_kwh": total_curtailment,
            "direct_consumption_rate": self._safe_div(total_direct, total_generation),
            "load_side_green_consumption_rate": self._safe_div(total_consumed, total_generation),
            "local_consumption_rate_including_storage_loss": self._safe_div(total_local_including_loss, total_generation),
            "green_power_ratio_of_load": self._safe_div(total_consumed, total_load),
            "export_rate": self._safe_div(total_export, total_generation),
            "grid_purchase_ratio_of_load": self._safe_div(total_purchase, total_load),
            "renewable_utilization_rate": renewable_utilization,
            "storage_final_soc_percent": float(df["storage_soc_percent"].iloc[-1]) if len(df) else self.initial_soc_percent,
            "diagnostics": list(dict.fromkeys(self.diagnostics)),
        }
