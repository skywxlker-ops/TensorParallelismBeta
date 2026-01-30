# Dataset Throughput Note

The **EdufineWeb** dataset runs slightly faster than **Bluwerp** for three main reasons:

1. **Fewer Files**: EdufineWeb has 3 shards, while Bluwerp has 5. Fewer files mean less time spent switching and loading data.
2. **Memory Cache**: EdufineWeb is smaller (~500MB), so it stays entirely in the computer's fast memory (RAM cache). Bluwerp is larger (~1GB), causing more slow disk reading.
3. **Warm-up Time**: The training starts slower while the system "warms up" its memory pool. Since Bluwerp logs started from a fresh run, those initial slow steps are more noticeable.

**In simple terms:** EdufineWeb is smaller and simpler, so the computer can keep it "ready" in fast memory more easily.
