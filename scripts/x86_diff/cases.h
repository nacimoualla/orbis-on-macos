// Shared by the native (Rosetta) and emulator harnesses.
#include <stdint.h>
struct Case { const char* name; uint8_t len; uint8_t bytes[12]; int mem; };
static const struct Case kCases[] = {
  {"vpaddq x5,x4,x2", 4, {0xc5,0xd9,0xd4,0xea}, 0},
  {"vmovq rsi,x4",    5, {0xc4,0xe1,0xf9,0x7e,0xe6}, 0},
  {"vpaddq x4,x4,x0", 4, {0xc5,0xd9,0xd4,0xe0}, 0},
  {"vpblendw x5,x5,x3,cc", 6, {0xc4,0xe3,0x51,0x0e,0xeb,0xcc}, 0},
  {"vpsrlq x5,3",     5, {0xc5,0xd1,0x73,0xd5,0x03}, 0},
  {"vpshufd x5,x5,e8",5, {0xc5,0xf9,0x70,0xed,0xe8}, 0},
  {"vmovq [rax+4rsi+1ab0],x5", 9, {0xc5,0xf9,0xd6,0xac,0xb0,0xb0,0x1a,0x00,0x00}, 1},
  {"vmovd x2,ecx",    4, {0xc5,0xf9,0x6e,0xd1}, 0},
  {"vpxor x3,x3,x3",  4, {0xc5,0xe1,0xef,0xdb}, 0},
  {"vpshufd x2,x2,44",5, {0xc5,0xf9,0x70,0xd2,0x44}, 0},
  {"vpand x7,x7,x5",  4, {0xc5,0xc1,0xdb,0xfd}, 0},
  {"vpaddq x7,x6,x2", 4, {0xc5,0xc9,0xd4,0xfa}, 0},
  {"vpblendw x7,x7,x3,cc", 6, {0xc4,0xe3,0x41,0x0e,0xfb,0xcc}, 0},
  {"vpsrlq x7,3",     5, {0xc5,0xc1,0x73,0xd7,0x03}, 0},
};
#define NCASES (sizeof(kCases)/sizeof(kCases[0]))
struct State { uint8_t ymm[16][32]; uint64_t rax, rcx, rdx, rsi; uint8_t mem[16]; };
static uint64_t rng_s;
static uint64_t rnd(void){ rng_s ^= rng_s<<13; rng_s ^= rng_s>>7; rng_s ^= rng_s<<17; return rng_s; }
static void fill(struct State* s, unsigned c, unsigned t){
  rng_s = 0x9E3779B97F4A7C15ull ^ (c*7919u + t*104729u + 1);
  for (int r=0;r<16;r++) for (int b=0;b<32;b+=8){ uint64_t v=rnd(); for(int k=0;k<8;k++) s->ymm[r][b+k]=(uint8_t)(v>>(8*k)); }
  s->rax=rnd(); s->rcx=rnd(); s->rdx=rnd(); s->rsi=rnd()&0xff;
  for(int k=0;k<16;k++) s->mem[k]=(uint8_t)rnd();
}
static void print_state(const struct State* s, unsigned c, unsigned t, int(*pr)(const char*,...)){
  pr("%u.%u", c, t);
  for (int r=0;r<16;r++){ pr(" y%d=",r); for(int b=31;b>=0;b--) pr("%02x",s->ymm[r][b]); }
  pr(" rax=%llx rcx=%llx rdx=%llx rsi=%llx mem=", (unsigned long long)s->rax,(unsigned long long)s->rcx,(unsigned long long)s->rdx,(unsigned long long)s->rsi);
  for(int k=0;k<16;k++) pr("%02x",s->mem[k]);
  pr("\n");
}
