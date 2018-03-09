# Face detection & alignment with MTCNN-PyCaffe

I wrapped the code of [pycaffe-mtcnn](https://github.com/kuangliu/pycaffe-mtcnn) to make it easier to use, and some tools for aligment is provided as well.

## Example code

```
if __name__ == '__main__':
    detector = MtcnnDetector()
    # Load image.
    im = cv2.imread(sys.argv[1])
    bboxes,points = detector.detect_face(im)
    #draw_and_show(im, bboxes, points)
    aligned = alignface_96x112(im, points)
    cv2.imwrite('align.png', aligned[0])
```

##  Alignment

The `alignment` module provide functions to compute the `affine` and `similarity` transform matrix.

	compute_affine_transform
	compute_similarity_transform
