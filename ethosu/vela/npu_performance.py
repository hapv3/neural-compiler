# SPDX-FileCopyrightText: Copyright 2020-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description:
# Simplified PassCycles enum for reporting performance metrics.

from enum import auto
from enum import IntEnum

class PassCycles(IntEnum):
    Npu = 0
    SramAccess = auto()
    DramAccess = auto()
    OnChipFlashAccess = auto()
    OffChipFlashAccess = auto()
    Total = auto()
    Size = auto()

    def display_name(self):
        return (
            "NPU",
            "SRAM Access",
            "DRAM Access",
            "On-chip Flash Access",
            "Off-chip Flash Access",
            "Total",
            "Size",
        )[self.value]

    def identifier_name(self):
        return (
            "npu",
            "sram_access",
            "dram_access",
            "on_chip_flash_access",
            "off_chip_flash_access",
            "total",
            "size",
        )[self.value]

    @staticmethod
    def all():
        return (
            PassCycles.Npu,
            PassCycles.SramAccess,
            PassCycles.DramAccess,
            PassCycles.OnChipFlashAccess,
            PassCycles.OffChipFlashAccess,
            PassCycles.Total,
        )
