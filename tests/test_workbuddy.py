"""WorkBuddy 接入测试（不触网、不调用 WorkBuddy.exe，只测纯函数/解析逻辑）。

覆盖：
- region 判定（workbuddy.ai / 子域 → global，其余 → cn）
- AAD 构造（_build_aad 字节布局）
- AES-256-GCM 解密（造一个已知明文信封，走 _decrypt_field 还原）
- 模型目录解析（id/name/credits/description/maxInputTokens/isDefault）
- 积分解析（CapacityRemain/Size/Used、TotalDosage）
- 请求头构造（commonHeaders + identityHeaders）
"""
from __future__ import annotations

import base64
import json
import struct

from quant.llm import workbuddy as wb


# ---------------- region ----------------

def test_region_global_domains():
    assert wb.WorkBuddyCredential("t", "r", domain="workbuddy.ai").region == "global"
    assert wb.WorkBuddyCredential("t", "r", domain="api.workbuddy.ai").region == "global"


def test_region_cn_domains():
    assert wb.WorkBuddyCredential("t", "r", domain="www.workbuddy.cn").region == "cn"
    assert wb.WorkBuddyCredential("t", "r", domain="workbuddy.cn").region == "cn"
    assert wb.WorkBuddyCredential("t", "r", domain="copilot.tencent.com").region == "cn"


# ---------------- AAD ----------------

def _len_prefix(v: bytes) -> bytes:
    return struct.pack(">I", len(v)) + v


def test_build_aad_layout():
    aad = wb._build_aad("0123456789abcdef", 1)
    expected = (b"WB-AAD\x00" + b"\x01"
                + _len_prefix(b"WBEV1") + _len_prefix(b"sym-v1")
                + struct.pack(">I", 1) + _len_prefix(b"0123456789abcdef")
                + b"\x02\x00\x00")
    assert aad == expected


# ---------------- AES 解密 ----------------

def test_decrypt_field_roundtrip():
    import hashlib

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    at_rest = "k" * 44
    protector_key = hashlib.sha256(at_rest.encode("utf-8")).digest()

    key_id = "0123456789abcdef"
    suite = 1
    aad = wb._build_aad(key_id, suite)

    nonce = b"n" * 12
    plaintext = b"hello-workbuddy-token"
    aesgcm = AESGCM(protector_key)
    ct = aesgcm.encrypt(nonce, plaintext, aad)
    ciphertext, auth_tag = ct[:-16], ct[-16:]

    envelope = {
        "suite": suite,
        "keyId": key_id,
        "nonce": base64.b64encode(nonce).decode(),
        "authTag": base64.b64encode(auth_tag).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }
    wrapped = {"$wbEncrypted": 1, "envelope": base64.b64encode(
        json.dumps(envelope).encode()).decode()}

    out = wb._decrypt_field(at_rest, wrapped)
    assert out == plaintext.decode("utf-8")


def test_decrypt_field_missing_envelope_raises():
    import pytest

    with pytest.raises(wb.WorkBuddyError):
        wb._decrypt_field("x", {"$wbEncrypted": 1})


# ---------------- 模型解析 ----------------

def test_parse_models_cn_structure():
    data = {
        "data": {"models": [
            {"id": "auto", "name": "Auto", "isDefault": True},
            {"id": "glm-5.3-flash", "name": "GLM-5.3 Flash", "credits": "x0.06",
             "descriptionZh": "快模型", "maxInputTokens": 128000,
             "maxOutputTokens": 8192, "isDefault": False},
        ]}
    }
    out = wb.WorkBuddyClient._parse_models(data)
    assert len(out) == 2
    assert out[0]["id"] == "auto"
    assert out[0]["credits"] is None
    assert out[0]["is_default"] is True
    assert out[1]["id"] == "glm-5.3-flash"
    assert out[1]["credits"] == "x0.06"
    assert out[1]["description"] == "快模型"
    assert out[1]["max_input_tokens"] == 128000


def test_parse_models_alternate_shapes():
    # ModelList 形式（老结构）
    data = {"data": {"Response": {"Data": {"ModelList": [
        {"ModelId": "a", "ModelName": "A", "Credits": "x1"}]}}}}
    out = wb.WorkBuddyClient._parse_models(data)
    assert out and out[0]["id"] == "a" and out[0]["name"] == "A"


# ---------------- 积分解析 ----------------

def test_parse_credits():
    data = {
        "data": {"Response": {"Data": {
            "TotalDosage": 4272,
            "Accounts": [
                {"PackageName": "拉新权益包", "CapacityRemain": 872, "CapacitySize": 1500,
                 "CapacityUsed": 628, "CapacityUnit": "credits",
                 "CycleEndTime": "2026-10-24 14:06:01"},
            ],
        }}}
    }
    out = wb.WorkBuddyClient._parse_credits(data)
    assert out["total"] == 4272
    assert out["accounts"][0]["name"] == "拉新权益包"
    assert out["accounts"][0]["remain"] == 872
    assert out["accounts"][0]["size"] == 1500
    assert out["accounts"][0]["used"] == 628
    assert out["accounts"][0]["unit"] == "credits"


# ---------------- 请求头 ----------------

def test_headers():
    cred = wb.WorkBuddyCredential(access_token="tok", refresh_token="rt", uid="u1",
                                  domain="www.workbuddy.cn", nickname="n")
    common = wb._common_headers(cred, "https://www.codebuddy.cn")
    assert common["X-Requested-With"] == "XMLHttpRequest"
    assert common["Origin"] == "https://www.codebuddy.cn"
    assert common["Referer"] == "https://www.codebuddy.cn/"
    assert common["User-Agent"] == wb.CLIENT_UA

    ident = wb._identity_headers(cred)
    assert ident["Authorization"] == "Bearer tok"
    assert ident["X-User-Id"] == "u1"
    assert ident["X-Domain"] == "www.workbuddy.cn"
