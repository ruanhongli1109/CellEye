"""培养箱环境传感器。

设计原则：**任何一路传感器失败都不能拖垮整条链路**。
每个读取都单独 try/except，失败就返回 None，最终由 pipeline 记录成空值。
真实培养场景里，探头松脱、I2C 总线被拉低是常态，不能让它中断采集。

- SHT31  温湿度
- SCD40  CO2（NDIR，需要单次测量命令）
- BMP280 气压/温度

默认走 mock，无需硬件。
"""

from __future__ import annotations

import math
import time
from datetime import datetime

from .config import Config
from .schema import SensorReading

# ---- SHT31 ----
SHT31_MEASURE_HIGHREP = 0x2C06

# ---- SCD40 ----
SCD40_MEASURE_SINGLE_SHOT = 0xEC05
SCD40_READ_INTERVAL = 5.0  # 秒，单次测量后需要等待


def _crc8_scd4x(data: bytes) -> int:
    """SCD4x 系列的 CRC-8 校验（多项式 0x31，初值 0xFF）。"""
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class MockSensorHub:
    """仿真传感器：围绕基线做缓慢漂移 + 少量噪声。"""

    def __init__(self, cfg: Config):
        self.base_t = float(cfg.get("sensors.baseline.temperature_c", 30.0))
        self.base_h = float(cfg.get("sensors.baseline.humidity_pct", 80.0))
        self.base_c = float(cfg.get("sensors.baseline.co2_ppm", 400.0))
        self.n = 0

    def read(self) -> SensorReading:
        i = self.n
        self.n += 1
        t = self.base_t + 0.25 * math.sin(i / 5.0)
        h = self.base_h + 1.2 * math.sin(i / 7.0 + 0.6)
        c = self.base_c + 18.0 * math.sin(i / 9.0)
        return SensorReading(
            timestamp=datetime.now(),
            temperature_c=round(t, 2),
            humidity_pct=round(h, 2),
            co2_ppm=round(c, 1),
            pressure_hpa=round(1013.2 + 0.4 * math.sin(i / 11.0), 2),
        )


class I2CSensorHub:
    """真实 I2C 传感器。任一传感器缺失时对应字段为 None。"""

    def __init__(self, cfg: Config):
        from smbus2 import SMBus  # 仅在树莓派上才需要

        self.bus = SMBus(int(cfg.get("sensors.i2c_bus", 1)))
        self.addr_sht31 = int(cfg.get("sensors.addresses.sht31", 0x44))
        self.addr_scd40 = int(cfg.get("sensors.addresses.scd40", 0x62))
        self.addr_bmp280 = int(cfg.get("sensors.addresses.bmp280", 0x76))
        self._bmp_calib = self._load_bmp280_calibration()

    # ---- SHT31 ----
    def _read_sht31(self) -> tuple[float | None, float | None]:
        self.bus.write_i2c_block_data(self.addr_sht31, SHT31_MEASURE_HIGHREP >> 8,
                                      [SHT31_MEASURE_HIGHREP & 0xFF])
        time.sleep(0.02)
        d = self.bus.read_i2c_block_data(self.addr_sht31, 0x00, 6)
        raw_t = (d[0] << 8) | d[1]
        raw_h = (d[3] << 8) | d[4]
        t = -45.0 + 175.0 * raw_t / 65535.0
        h = 100.0 * raw_h / 65535.0
        return round(t, 2), round(h, 2)

    # ---- SCD40 ----
    def _read_scd40(self) -> tuple[float | None, float | None, float | None]:
        self.bus.write_i2c_block_data(self.addr_scd40, SCD40_MEASURE_SINGLE_SHOT >> 8,
                                      [SCD40_MEASURE_SINGLE_SHOT & 0xFF])
        time.sleep(SCD40_READ_INTERVAL)
        d = self.bus.read_i2c_block_data(self.addr_scd40, 0x00, 9)
        co2 = (d[0] << 8) | d[1]
        raw_t = (d[3] << 8) | d[4]
        raw_h = (d[6] << 8) | d[7]
        # 校验不通过就丢弃，避免把噪声当数据
        if _crc8_scd4x(bytes(d[0:2])) != d[2]:
            co2 = None
        t = -45.0 + 175.0 * raw_t / 65535.0
        h = 100.0 * raw_h / 65535.0
        return (None if co2 is None else float(co2)), round(t, 2), round(h, 2)

    # ---- BMP280 ----
    def _load_bmp280_calibration(self) -> dict[str, int] | None:
        try:
            d = self.bus.read_i2c_block_data(self.addr_bmp280, 0x88, 24)
            u16 = lambda i: d[i] | (d[i + 1] << 8)  # noqa: E731
            s16 = lambda i: (d[i] | (d[i + 1] << 8)) - 65536 if d[i + 1] & 0x80 else d[i] | (d[i + 1] << 8)  # noqa: E731
            return {
                "T1": u16(0), "T2": s16(2), "T3": s16(4),
                "P1": u16(6), "P2": s16(8), "P3": s16(10), "P4": s16(12),
                "P5": s16(14), "P6": s16(16), "P7": s16(18), "P8": s16(20), "P9": s16(22),
            }
        except Exception:
            return None

    def _read_bmp280(self) -> float | None:
        c = self._bmp_calib
        if not c:
            return None
        # 强制一次测量：温压各 oversampling x1，正常模式
        self.bus.write_i2c_block_data(self.addr_bmp280, 0xF4, [0x27])
        time.sleep(0.05)
        d = self.bus.read_i2c_block_data(self.addr_bmp280, 0xF7, 6)
        adc_p = (d[0] << 12) | (d[1] << 4) | (d[2] >> 4)
        adc_t = (d[3] << 12) | (d[4] << 4) | (d[5] >> 4)

        # 温度补偿（BMP280 数据手册）
        var1 = ((adc_t / 16384.0) - (c["T1"] / 1024.0)) * c["T2"]
        var2 = ((adc_t / 131072.0) - (c["T1"] / 8192.0)) ** 2 * c["T3"]
        t_fine = var1 + var2

        # 气压补偿
        var1 = t_fine / 2.0 - 64000.0
        var2 = var1 * var1 * c["P6"] / 32768.0
        var2 = var2 + var1 * c["P5"] * 2.0
        var2 = var2 / 4.0 + c["P4"] * 65536.0
        var1 = (c["P3"] * var1 * var1 / 524288.0 + c["P2"] * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * c["P1"]
        if var1 == 0:
            return None
        p = 1048576.0 - adc_p
        p = (p - var2 / 4096.0) * 6250.0 / var1
        var1 = c["P9"] * p * p / 2147483648.0
        var2 = p * c["P8"] / 32768.0
        p = p + (var1 + var2 + c["P7"]) / 16.0
        return round(p / 100.0, 2)  # Pa -> hPa

    def read(self) -> SensorReading:
        t = h = co2 = pres = None
        try:
            t, h = self._read_sht31()
        except Exception:
            pass
        try:
            co2, t2, h2 = self._read_scd40()
            t = t if t is not None else t2
            h = h if h is not None else h2
        except Exception:
            pass
        try:
            pres = self._read_bmp280()
        except Exception:
            pass
        return SensorReading(timestamp=datetime.now(), temperature_c=t,
                             humidity_pct=h, co2_ppm=co2, pressure_hpa=pres)


def make_sensor_hub(cfg: Config):
    if not bool(cfg.get("sensors.enabled", True)):
        return None
    if bool(cfg.get("sensors.mock", True)):
        return MockSensorHub(cfg)
    return I2CSensorHub(cfg)
