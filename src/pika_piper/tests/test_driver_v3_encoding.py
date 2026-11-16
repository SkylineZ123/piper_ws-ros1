"""Check real SDK wire encoding without opening CAN or constructing an arm."""
import pytest


def test_high_follow_v3_mode_frame():
    protocol=pytest.importorskip('piper_sdk.protocol.protocol_v3')
    from piper_sdk.piper_msgs.msg_v3 import PiperMessage,ArmMsgType,ArmMsgMotionCtrl_2
    from can import Message
    frame=Message()
    protocol.C_PiperParserV3().EncodeMessage(PiperMessage(type_=ArmMsgType.PiperMsgMotionCtrl_2,
        arm_motion_ctrl_2=ArmMsgMotionCtrl_2(1,1,100,0xAD)),frame)
    assert frame.arbitration_id==0x151
    assert bytes(frame.data)==bytes([1,1,100,0xAD,0,0,0,0])


def test_speed_only_frame_keeps_angle_fields_invalid():
    protocol=pytest.importorskip('piper_sdk.protocol.protocol_v3')
    from piper_sdk.piper_msgs.msg_v3 import PiperMessage,ArmMsgType,ArmMsgMotorAngleLimitMaxSpdSet
    from can import Message
    frame=Message()
    protocol.C_PiperParserV3().EncodeMessage(PiperMessage(type_=ArmMsgType.PiperMsgMotorAngleLimitMaxSpdSet,
        arm_motor_angle_limit_max_spd_set=ArmMsgMotorAngleLimitMaxSpdSet(1,0x7FFF,0x7FFF,1500)),frame)
    assert frame.arbitration_id==0x474
    assert bytes(frame.data)==bytes([1,0x7F,0xFF,0x7F,0xFF,0x05,0xDC,0])
