def mergeFingerprints(fps, fmap, threshold=0.3):
    merged_fps = []
    merged_fmap = {}

    processed = []
    for ai, afp in enumerate(fps):
        # skip fingerprints that have already been merged
        if ai in processed: continue

        # get best scoring fingerprint
        score, bi, bfp = sorted([(getSimilarity(afp, bfp), bi, bfp) for bi, bfp in enumerate(fps)]).pop()

        # same for second fingerprint
        if bi in processed: continue

        if score > threshold:
            # merge fingerprints
            fp = updateFingerprint(afp, bfp, 2)
            merged_fps.append(fp)
            # merge node references
            i = len(merged_fps) - 1
            merged_fmap[i] = fmap[ai] + fmap[bi]
            # mark as processed
            processed += [ai, bi]

    return merged_fps, merged_fmap