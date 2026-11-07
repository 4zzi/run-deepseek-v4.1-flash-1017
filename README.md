Run DeepSeek "V4.1 Flash 1017" using a gaming PC and Strata's SSD streaming

Specs used (no less):

```
RTX 5060 TI 8gb VRAM

Ryzen 5 7600X

16gb of DDR5

1TB of Kingston NV3 NVME
```

Run the start.sh in your linux environment it has a tune thing, make sure to change stuff in there to point to the models GGUF

Stats for this spec (drive speed of 2.5gb):
```
1M window context token

6 experts 

Vision on

100 token/s decode

3 token/s writing speed
maybe 2-5 more if you have a faster drive (6gb/s)
```

You would be pretty comfortable if you had 32gb of ram and a 3060 12gb

Note: 

this is IstaDash lab's GCO RCQ Coder version at Q2XXS

if you look at the models reasoning, it's a little impaired and sounds like shakespeare ("need not overdo", "need concise") 

but the response it writes is somewhat fine

can hallucinate and swirl into a loop

run if you want
