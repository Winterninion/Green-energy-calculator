import io
import math
from typing import Any, Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from core_engine import PVConsumptionCalculator

app = FastAPI(title="风光储绿电直连消纳测算 API")

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/index.html")
async def index():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if not isinstance(obj, (str, bytes, bytearray)):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
    return obj


def pct(x: float) -> float:
    return round(float(x) * 100.0, 2)


def kwh(x: float) -> float:
    return round(float(x), 2)


def wan_kwh(x: float) -> float:
    return round(float(x) / 10000.0, 2)


def build_extended_indicators(summary: dict) -> dict:
    """生成申报指标扩展版。所有电量统一展示为“万kWh”。"""
    total_load = float(summary["total_load_kwh"])
    net_load = float(summary["net_load_after_internal_storage_kwh"])
    total_generation = float(summary["total_generation_kwh"])
    self_use = float(summary["total_consumed_kwh"])
    grid_export = float(summary["total_grid_export_kwh"])
    grid_purchase = float(summary["total_grid_purchase_kwh"])
    curtailment = float(summary["total_curtailment_kwh"])

    self_use_generation_ratio = self_use / total_generation if total_generation > 0 else 0.0
    self_use_load_ratio = self_use / total_load if total_load > 0 else 0.0
    export_rate = grid_export / total_generation if total_generation > 0 else 0.0
    renewable_utilization_rate = summary.get("renewable_utilization_rate", 0.0)

    rows = [
        {"name": "企业年用电量", "value": f"{wan_kwh(total_load)} 万kWh", "value_wan_kwh": wan_kwh(total_load), "value_kwh": kwh(total_load), "unit": "kWh"},
        {"name": "扣除企业内部光储调节后年用电量", "value": f"{wan_kwh(net_load)} 万kWh", "value_wan_kwh": wan_kwh(net_load), "value_kwh": kwh(net_load), "unit": "kWh"},
        {"name": "企业负荷年平均负载率", "value": f"{pct(summary['average_load_rate'])}%", "value_percent": pct(summary["average_load_rate"]), "unit": "%"},
        {"name": "风电规模", "value": f"{round(float(summary.get('wind_capacity_mw', 0.0)), 4):g} MW", "value_mw": round(float(summary.get("wind_capacity_mw", 0.0)), 4), "unit": "MW"},
        {"name": "新能源年发电量", "value": f"{wan_kwh(total_generation)} 万kWh", "value_wan_kwh": wan_kwh(total_generation), "value_kwh": kwh(total_generation), "unit": "kWh"},
        {"name": "年自发自用电量", "value": f"{wan_kwh(self_use)} 万kWh", "value_wan_kwh": wan_kwh(self_use), "value_kwh": kwh(self_use), "unit": "kWh"},
        {"name": "年自发自用电量占总可用发电量比例", "value": f"{pct(self_use_generation_ratio)}%", "value_percent": pct(self_use_generation_ratio), "unit": "%"},
        {"name": "年自发自用电量占总用电量比例", "value": f"{pct(self_use_load_ratio)}%", "value_percent": pct(self_use_load_ratio), "unit": "%"},
        {"name": "年上网电量", "value": f"{wan_kwh(grid_export)} 万kWh", "value_wan_kwh": wan_kwh(grid_export), "value_kwh": kwh(grid_export), "unit": "kWh"},
        {"name": "上网电量占比", "value": f"{pct(export_rate)}%", "value_percent": pct(export_rate), "unit": "%"},
        {"name": "年弃电量", "value": f"{wan_kwh(curtailment)} 万kWh", "value_wan_kwh": wan_kwh(curtailment), "value_kwh": kwh(curtailment), "unit": "kWh"},
        {"name": "新能源利用率", "value": f"{pct(renewable_utilization_rate)}%", "value_percent": pct(renewable_utilization_rate), "unit": "%"},
        {"name": "与电网年交换电量（下网）", "value": f"{wan_kwh(grid_purchase)} 万kWh", "value_wan_kwh": wan_kwh(grid_purchase), "value_kwh": kwh(grid_purchase), "unit": "kWh"},
    ]

    return {
        "rows": rows,
        "raw": {
            "enterprise_annual_load_kwh": kwh(total_load),
            "net_load_after_internal_storage_kwh": kwh(net_load),
            "average_load_rate": pct(summary["average_load_rate"]),
            "wind_capacity_mw": round(float(summary.get("wind_capacity_mw", 0.0)), 4),
            "new_energy_generation_kwh": kwh(total_generation),
            "self_use_kwh": kwh(self_use),
            "self_use_generation_ratio": pct(self_use_generation_ratio),
            "self_use_load_ratio": pct(self_use_load_ratio),
            "grid_export_kwh": kwh(grid_export),
            "export_rate": pct(export_rate),
            "curtailment_kwh": kwh(curtailment),
            "renewable_utilization_rate": pct(renewable_utilization_rate),
            "grid_purchase_kwh": kwh(grid_purchase),
        },
    }


@app.post("/api/calculate")
async def calculate_endpoint(
    capacity_mw: float = Form(10.6),
    wind_capacity_mw: float = Form(0.0),
    wind_target_hours: float = Form(0.0),
    grid_capacity_mva: float = Form(15.0),
    fixed_load_mw: float = Form(0.0),
    # 新版前端使用 load_files 支持任意多个负荷文件；保留 load_file 兼容旧前端。
    load_files: Optional[List[UploadFile]] = File(None),
    load_file: Optional[UploadFile] = File(None),
    pv_file: Optional[UploadFile] = File(None),
    wind_file: Optional[UploadFile] = File(None),
    enable_storage: bool = Form(True),
    storage_power_mw: float = Form(5.0),
    storage_energy_mwh: float = Form(10.264),
    charge_efficiency: float = Form(95.0),
    discharge_efficiency: float = Form(95.0),
    min_soc_percent: float = Form(5.0),
    max_soc_percent: float = Form(100.0),
    initial_soc_percent: float = Form(5.0),
):
    try:
        upload_loads: list[UploadFile] = []
        if load_files:
            upload_loads.extend([f for f in load_files if f is not None and f.filename])
        if load_file is not None and load_file.filename:
            upload_loads.append(load_file)
        if not upload_loads:
            raise ValueError("请至少上传 1 个用电负荷表。")

        load_bytes_list = []
        for i, lf in enumerate(upload_loads, start=1):
            b = io.BytesIO(await lf.read())
            b.name = lf.filename or f"load_file_{i}"
            load_bytes_list.append(b)

        pv_bytes = None
        if pv_file is not None and pv_file.filename:
            pv_bytes = io.BytesIO(await pv_file.read())
            pv_bytes.name = pv_file.filename or "pv_file"

        wind_bytes = None
        if wind_file is not None and wind_file.filename:
            wind_bytes = io.BytesIO(await wind_file.read())
            wind_bytes.name = wind_file.filename or "wind_file"

        calculator = PVConsumptionCalculator(
            pv_capacity_mw=capacity_mw,
            wind_capacity_mw=wind_capacity_mw,
            wind_target_hours=wind_target_hours,
            grid_capacity_mva=grid_capacity_mva,
            fixed_load_mw=fixed_load_mw,
            storage_power_mw=storage_power_mw,
            storage_energy_mwh=storage_energy_mwh,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            min_soc_percent=min_soc_percent,
            max_soc_percent=max_soc_percent,
            initial_soc_percent=initial_soc_percent,
            enable_storage=enable_storage,
        )
        summary, df_monthly, df_detail = calculator.calculate(load_bytes_list, pv_bytes, wind_bytes)
        df_monthly_reset = df_monthly.reset_index()
        standard_indicators = build_extended_indicators(summary)

        response_data = {
            "success": True,
            "params": {
                "capacity_mw": float(capacity_mw),
                "wind_capacity_mw": float(wind_capacity_mw),
                "wind_target_hours": float(wind_target_hours),
                "grid_capacity_mva": float(grid_capacity_mva),
                "fixed_load_mw": float(fixed_load_mw),
                "load_file_count": len(load_bytes_list),
                "enable_storage": bool(enable_storage),
                "storage_power_mw": float(storage_power_mw),
                "storage_energy_mwh": float(storage_energy_mwh),
                "charge_efficiency": float(charge_efficiency),
                "discharge_efficiency": float(discharge_efficiency),
                "min_soc_percent": float(min_soc_percent),
                "max_soc_percent": float(max_soc_percent),
                "initial_soc_percent": float(initial_soc_percent),
            },
            "standard_indicators": standard_indicators,
            "diagnostics": summary.get("diagnostics", []),
            # 兼容旧前端字段，同时新增风电/新能源字段。
            "annual_rate": standard_indicators["raw"]["self_use_generation_ratio"],
            "green_power_ratio_of_load": standard_indicators["raw"]["self_use_load_ratio"],
            "export_rate": standard_indicators["raw"]["export_rate"],
            "grid_purchase_ratio_of_load": pct(summary["grid_purchase_ratio_of_load"]),
            "storage_final_soc_percent": round(float(summary["storage_final_soc_percent"]), 2),
            "total_pv_kwh": kwh(summary["total_pv_kwh"]),
            "total_wind_kwh": kwh(summary["total_wind_kwh"]),
            "wind_equivalent_hours_actual": round(float(summary.get("wind_equivalent_hours_actual", 0.0)), 2),
            "wind_target_hours": round(float(summary.get("wind_target_hours", 0.0)), 2),
            "total_generation_kwh": kwh(summary["total_generation_kwh"]),
            "total_load_kwh": kwh(summary["total_load_kwh"]),
            "total_uploaded_load_kwh": kwh(summary.get("total_uploaded_load_kwh", summary["total_load_kwh"])),
            "total_fixed_load_kwh": kwh(summary.get("total_fixed_load_kwh", 0.0)),
            "fixed_load_mw": float(summary.get("fixed_load_mw", fixed_load_mw)),
            "net_load_after_internal_storage_kwh": kwh(summary["net_load_after_internal_storage_kwh"]),
            "average_load_rate": pct(summary["average_load_rate"]),
            "max_load_curve_count": int(round(float(summary.get("max_load_curve_count", 1)))),
            "avg_load_curve_count": round(float(summary.get("avg_load_curve_count", 1)), 2),
            "max_active_load_curve_count": int(round(float(summary.get("max_active_load_curve_count", 1)))),
            "total_consumed_kwh": kwh(summary["total_consumed_kwh"]),
            "total_direct_consumed_kwh": kwh(summary["total_direct_consumed_kwh"]),
            "total_storage_charge_kwh": kwh(summary["total_storage_charge_kwh"]),
            "total_storage_discharge_kwh": kwh(summary["total_storage_discharge_kwh"]),
            "total_storage_loss_kwh": kwh(summary["total_storage_loss_kwh"]),
            "total_grid_export_kwh": kwh(summary["total_grid_export_kwh"]),
            "total_grid_purchase_kwh": kwh(summary["total_grid_purchase_kwh"]),
            "total_curtailment_kwh": kwh(summary["total_curtailment_kwh"]),
            "renewable_utilization_rate": pct(summary["renewable_utilization_rate"]),
            "monthly_data": {
                "months": df_monthly_reset["month"].astype(int).tolist(),
                "pv_kw": df_monthly_reset["pv_total_kw"].astype(float).round(2).tolist(),
                "wind_kw": df_monthly_reset["wind_total_kw"].astype(float).round(2).tolist(),
                "renewable_kw": df_monthly_reset["renewable_total_kw"].astype(float).round(2).tolist(),
                "load_kw": df_monthly_reset["load_kw"].astype(float).round(2).tolist(),
                "uploaded_load_kw": df_monthly_reset.get("uploaded_load_kw", df_monthly_reset["load_kw"]).astype(float).round(2).tolist(),
                "fixed_load_kw": df_monthly_reset.get("fixed_load_kw", pd.Series([0.0] * len(df_monthly_reset))).astype(float).round(2).tolist(),
                "load_curve_count_avg": df_monthly_reset.get("load_curve_count_avg", pd.Series([1] * len(df_monthly_reset))).astype(float).round(2).tolist(),
                "load_curve_count_max": df_monthly_reset.get("load_curve_count_max", pd.Series([1] * len(df_monthly_reset))).astype(float).round(2).tolist(),
                "consumed_kw": df_monthly_reset["consumed_kw"].astype(float).round(2).tolist(),
                "grid_export_kw": df_monthly_reset["grid_export_kw"].astype(float).round(2).tolist(),
                "grid_purchase_kw": df_monthly_reset["grid_purchase_kw"].astype(float).round(2).tolist(),
                "curtailment_kw": df_monthly_reset["curtailment_kw"].astype(float).round(2).tolist(),
                "self_use_generation_rates": np.where(
                    df_monthly_reset["renewable_total_kw"].astype(float) > 0,
                    df_monthly_reset["consumed_kw"].astype(float) / df_monthly_reset["renewable_total_kw"].astype(float) * 100,
                    0,
                ).round(2).tolist(),
                "self_use_load_rates": np.where(
                    df_monthly_reset["load_kw"].astype(float) > 0,
                    df_monthly_reset["consumed_kw"].astype(float) / df_monthly_reset["load_kw"].astype(float) * 100,
                    0,
                ).round(2).tolist(),
                "export_rates": np.where(
                    df_monthly_reset["renewable_total_kw"].astype(float) > 0,
                    df_monthly_reset["grid_export_kw"].astype(float) / df_monthly_reset["renewable_total_kw"].astype(float) * 100,
                    0,
                ).round(2).tolist(),
            },
            "soc_preview": {
                "hours": list(range(min(240, len(df_detail)))),
                "soc_percent": df_detail["storage_soc_percent"].iloc[:240].astype(float).round(2).tolist(),
            },
        }
        return JSONResponse(content=to_json_safe(response_data))

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
