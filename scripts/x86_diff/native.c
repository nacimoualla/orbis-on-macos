#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include "cases.h"
// Builds: vmovdqu all ymm from state; load gprs; <insn>; store back; ret.
static size_t emit(uint8_t* p, const struct Case* c){
  size_t n=0;
  for(int r=0;r<16;r++){ // vmovdqu ymmR,[rdi+32r]: C5/C4 forms -> use C4 E1/61 7E 6F
    uint8_t R = (r<8)?0x80:0x00; // VEX.R inverted
    p[n++]=0xC4; p[n++]=(uint8_t)(R|0x61); p[n++]=0x7E; p[n++]=0x6F; p[n++]=(uint8_t)(0x87|((r&7)<<3)); uint32_t d=32*r; memcpy(p+n,&d,4); n+=4; }
  const uint8_t ld[]={0x48,0x8b,0x87,0,2,0,0, 0x48,0x8b,0x8f,8,2,0,0, 0x48,0x8b,0x97,16,2,0,0, 0x48,0x8b,0xb7,24,2,0,0};
  memcpy(p+n,ld,sizeof ld); n+=sizeof ld;
  memcpy(p+n,c->bytes,c->len); n+=c->len;
  for(int r=0;r<16;r++){ uint8_t R=(r<8)?0x80:0x00; p[n++]=0xC4; p[n++]=(uint8_t)(R|0x61); p[n++]=0x7E; p[n++]=0x7F; p[n++]=(uint8_t)(0x87|((r&7)<<3)); uint32_t d=32*r; memcpy(p+n,&d,4); n+=4; }
  const uint8_t st[]={0x48,0x89,0x87,0,2,0,0, 0x48,0x89,0x8f,8,2,0,0, 0x48,0x89,0x97,16,2,0,0, 0x48,0x89,0xb7,24,2,0,0, 0xc5,0xf8,0x77, 0xc3};
  memcpy(p+n,st,sizeof st); n+=sizeof st; return n;
}
int main(void){
  uint8_t* code=mmap(0,4096,PROT_READ|PROT_WRITE|PROT_EXEC,MAP_PRIVATE|MAP_ANON|MAP_JIT,-1,0);
  if(code==MAP_FAILED){ perror("mmap"); return 1; }
  for(unsigned c=0;c<NCASES;c++){
    emit(code,&kCases[c]);
    for(unsigned t=0;t<8;t++){
      static struct State s; fill(&s,c,t);
      if(kCases[c].mem){ s.rax=(uint64_t)(uintptr_t)s.mem-0x1ab0; s.rsi=0; }
      uint64_t rax0=s.rax;
      ((void(*)(struct State*))code)(&s);
      if(kCases[c].mem){ s.rax-=rax0; }
      print_state(&s,c,t,printf);
    }
  }
}
