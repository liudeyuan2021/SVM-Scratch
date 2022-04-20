#ifndef SVM_INCLUDE_GUARD_SVM_01_SUPPORT_H
#define SVM_INCLUDE_GUARD_SVM_01_SUPPORT_H

#ifdef _MSC_VER
__declspec(align(4))
#else
__attribute__((aligned(4)))
#endif
static const unsigned char support_int32_bin[] = {
0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x05,0x00,0x00,0x00,0x08,0x00,0x00,0x00,
0x0d,0x00,0x00,0x00,0x10,0x00,0x00,0x00,0x11,0x00,0x00,0x00,0x16,0x00,0x00,0x00,
0x17,0x00,0x00,0x00,0x1e,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0x23,0x00,0x00,0x00,
0x27,0x00,0x00,0x00,0x2b,0x00,0x00,0x00,0x30,0x00,0x00,0x00,0x32,0x00,0x00,0x00,
0x33,0x00,0x00,0x00,0x34,0x00,0x00,0x00,0x39,0x00,0x00,0x00,0x3f,0x00,0x00,0x00,
0x41,0x00,0x00,0x00,0x43,0x00,0x00,0x00,0x4b,0x00,0x00,0x00,0x4f,0x00,0x00,0x00,
0x52,0x00,0x00,0x00,0x54,0x00,0x00,0x00,0x58,0x00,0x00,0x00,0x5d,0x00,0x00,0x00,
0x61,0x00,0x00,0x00,0x62,0x00,0x00,0x00,0x64,0x00,0x00,0x00,0x6a,0x00,0x00,0x00,
0x6b,0x00,0x00,0x00,0x73,0x00,0x00,0x00,0x75,0x00,0x00,0x00,0x7e,0x00,0x00,0x00,
0x80,0x00,0x00,0x00,0x85,0x00,0x00,0x00,0x8d,0x00,0x00,0x00,0x95,0x00,0x00,0x00,
0x96,0x00,0x00,0x00,0x9f,0x00,0x00,0x00,0xa2,0x00,0x00,0x00,0xa3,0x00,0x00,0x00,
0xa7,0x00,0x00,0x00,0xae,0x00,0x00,0x00,0xb0,0x00,0x00,0x00,0xb4,0x00,0x00,0x00,
0xb7,0x00,0x00,0x00,0xb9,0x00,0x00,0x00,0xbb,0x00,0x00,0x00,0xc0,0x00,0x00,0x00,
0xc2,0x00,0x00,0x00,0xc8,0x00,0x00,0x00,0xcd,0x00,0x00,0x00,0xce,0x00,0x00,0x00,
0xcf,0x00,0x00,0x00,0xd3,0x00,0x00,0x00,0xde,0x00,0x00,0x00,0xe0,0x00,0x00,0x00,
0xe4,0x00,0x00,0x00,0xe5,0x00,0x00,0x00,0xe7,0x00,0x00,0x00,0xe8,0x00,0x00,0x00,
0xe9,0x00,0x00,0x00,0xed,0x00,0x00,0x00,0xf2,0x00,0x00,0x00,0xf6,0x00,0x00,0x00,
0xfe,0x00,0x00,0x00,0xff,0x00,0x00,0x00,0x05,0x01,0x00,0x00,0x07,0x01,0x00,0x00,
0x08,0x01,0x00,0x00,0x0e,0x01,0x00,0x00,0x13,0x01,0x00,0x00,0x14,0x01,0x00,0x00,
0x19,0x01,0x00,0x00,0x21,0x01,0x00,0x00,0x22,0x01,0x00,0x00,0x23,0x01,0x00,0x00,
0x24,0x01,0x00,0x00,0x2b,0x01,0x00,0x00,0x31,0x01,0x00,0x00,0x32,0x01,0x00,0x00,
0x37,0x01,0x00,0x00,0x3e,0x01,0x00,0x00,0x44,0x01,0x00,0x00,0x4d,0x01,0x00,0x00,
0x4e,0x01,0x00,0x00,0x50,0x01,0x00,0x00,0x53,0x01,0x00,0x00,0x54,0x01,0x00,0x00,
0x5b,0x01,0x00,0x00,0x5c,0x01,0x00,0x00,0x61,0x01,0x00,0x00,0x6a,0x01,0x00,0x00,
0x6b,0x01,0x00,0x00,0x6c,0x01,0x00,0x00,0x72,0x01,0x00,0x00,0x7a,0x01,0x00,0x00,
0x7b,0x01,0x00,0x00,0x7d,0x01,0x00,0x00,0x7f,0x01,0x00,0x00,0x80,0x01,0x00,0x00,
0x82,0x01,0x00,0x00,0x85,0x01,0x00,0x00,0x87,0x01,0x00,0x00,0x8c,0x01,0x00,0x00,
0x95,0x01,0x00,0x00,0x97,0x01,0x00,0x00,0x9b,0x01,0x00,0x00,0xa3,0x01,0x00,0x00,
0xa9,0x01,0x00,0x00,0xaa,0x01,0x00,0x00,0xb7,0x01,0x00,0x00,0xc0,0x01,0x00,0x00,
0xc1,0x01,0x00,0x00,0xcd,0x01,0x00,0x00,0xd1,0x01,0x00,0x00,0xd8,0x01,0x00,0x00,
0xd9,0x01,0x00,0x00,0xda,0x01,0x00,0x00,0xe4,0x01,0x00,0x00,0xeb,0x01,0x00,0x00,
0xec,0x01,0x00,0x00,0xf6,0x01,0x00,0x00,0xf7,0x01,0x00,0x00,0xfb,0x01,0x00,0x00,
0xfd,0x01,0x00,0x00,0x02,0x02,0x00,0x00,0x10,0x02,0x00,0x00,0x15,0x02,0x00,0x00,
0x18,0x02,0x00,0x00,0x19,0x02,0x00,0x00,0x20,0x02,0x00,0x00,0x28,0x02,0x00,0x00,
0x2b,0x02,0x00,0x00,0x2e,0x02,0x00,0x00,0x2f,0x02,0x00,0x00,0x34,0x02,0x00,0x00,
0x35,0x02,0x00,0x00,0x36,0x02,0x00,0x00,0x3c,0x02,0x00,0x00,0x41,0x02,0x00,0x00,
0x44,0x02,0x00,0x00,0x45,0x02,0x00,0x00,0x48,0x02,0x00,0x00,0x49,0x02,0x00,0x00,
0x50,0x02,0x00,0x00,0x51,0x02,0x00,0x00,0x53,0x02,0x00,0x00,0x56,0x02,0x00,0x00,
0x59,0x02,0x00,0x00,0x5e,0x02,0x00,0x00,0x64,0x02,0x00,0x00,0x65,0x02,0x00,0x00,
0x66,0x02,0x00,0x00,0x69,0x02,0x00,0x00,0x6c,0x02,0x00,0x00,0x6d,0x02,0x00,0x00,
0x6e,0x02,0x00,0x00,0x72,0x02,0x00,0x00,0x77,0x02,0x00,0x00,0x78,0x02,0x00,0x00,
0x7b,0x02,0x00,0x00,0x85,0x02,0x00,0x00,0x88,0x02,0x00,0x00,0x89,0x02,0x00,0x00,
0x8a,0x02,0x00,0x00,0x8b,0x02,0x00,0x00,0x8e,0x02,0x00,0x00,0x91,0x02,0x00,0x00,
0x99,0x02,0x00,0x00,0x9a,0x02,0x00,0x00,0xa2,0x02,0x00,0x00,0xa7,0x02,0x00,0x00,
0xae,0x02,0x00,0x00,0xaf,0x02,0x00,0x00,0xb3,0x02,0x00,0x00,0xbb,0x02,0x00,0x00,
0xbc,0x02,0x00,0x00,0xc4,0x02,0x00,0x00,0xc6,0x02,0x00,0x00,0xcd,0x02,0x00,0x00,
0xd3,0x02,0x00,0x00,0xd4,0x02,0x00,0x00,0xd5,0x02,0x00,0x00,0xda,0x02,0x00,0x00,
0xdb,0x02,0x00,0x00,0xdd,0x02,0x00,0x00,0xde,0x02,0x00,0x00,0xe3,0x02,0x00,0x00,
0xe7,0x02,0x00,0x00,0xec,0x02,0x00,0x00,0xef,0x02,0x00,0x00,0xf4,0x02,0x00,0x00,
0xfb,0x02,0x00,0x00,0x19,0x00,0x00,0x00,0x1b,0x00,0x00,0x00,0x36,0x00,0x00,0x00,
0x3b,0x00,0x00,0x00,0x3e,0x00,0x00,0x00,0x44,0x00,0x00,0x00,0x46,0x00,0x00,0x00,
0x4a,0x00,0x00,0x00,0x4d,0x00,0x00,0x00,0x55,0x00,0x00,0x00,0x57,0x00,0x00,0x00,
0x5c,0x00,0x00,0x00,0x67,0x00,0x00,0x00,0x70,0x00,0x00,0x00,0x79,0x00,0x00,0x00,
0x7a,0x00,0x00,0x00,0x7f,0x00,0x00,0x00,0x86,0x00,0x00,0x00,0x8f,0x00,0x00,0x00,
0x9a,0x00,0x00,0x00,0xa6,0x00,0x00,0x00,0xad,0x00,0x00,0x00,0xd8,0x00,0x00,0x00,
0xee,0x00,0x00,0x00,0xfd,0x00,0x00,0x00,0x20,0x01,0x00,0x00,0x27,0x01,0x00,0x00,
0x29,0x01,0x00,0x00,0x2d,0x01,0x00,0x00,0x38,0x01,0x00,0x00,0x3c,0x01,0x00,0x00,
0x41,0x01,0x00,0x00,0x48,0x01,0x00,0x00,0x4b,0x01,0x00,0x00,0x52,0x01,0x00,0x00,
0x58,0x01,0x00,0x00,0x59,0x01,0x00,0x00,0x60,0x01,0x00,0x00,0x6d,0x01,0x00,0x00,
0x6e,0x01,0x00,0x00,0x6f,0x01,0x00,0x00,0x7c,0x01,0x00,0x00,0x8a,0x01,0x00,0x00,
0x9a,0x01,0x00,0x00,0xa8,0x01,0x00,0x00,0xaf,0x01,0x00,0x00,0xb3,0x01,0x00,0x00,
0xb5,0x01,0x00,0x00,0xc4,0x01,0x00,0x00,0xcb,0x01,0x00,0x00,0xce,0x01,0x00,0x00,
0xe2,0x01,0x00,0x00,0xe6,0x01,0x00,0x00,0xe7,0x01,0x00,0x00,0xef,0x01,0x00,0x00,
0xf2,0x01,0x00,0x00,0x00,0x02,0x00,0x00,0x09,0x02,0x00,0x00,0x0a,0x02,0x00,0x00,
0x0c,0x02,0x00,0x00,0x0f,0x02,0x00,0x00,0x11,0x02,0x00,0x00,0x14,0x02,0x00,0x00,
0x1a,0x02,0x00,0x00,0x1e,0x02,0x00,0x00,0x1f,0x02,0x00,0x00,0x31,0x02,0x00,0x00,
0x32,0x02,0x00,0x00,0x39,0x02,0x00,0x00,0x4b,0x02,0x00,0x00,0x4d,0x02,0x00,0x00,
0x5b,0x02,0x00,0x00,0x5c,0x02,0x00,0x00,0x5f,0x02,0x00,0x00,0x61,0x02,0x00,0x00,
0x63,0x02,0x00,0x00,0x9c,0x02,0x00,0x00,0xab,0x02,0x00,0x00,0xc5,0x02,0x00,0x00,
0xc7,0x02,0x00,0x00,0xfa,0x02,0x00,0x00,};
#endif // SVM_INCLUDE_GUARD_SVM_01_SUPPORT_H
