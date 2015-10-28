
/* a 3d volume class */

#ifndef VOLUME_H
#define VOLUME_H

template <class T>
class volume {
public:
    /* create a volume */
    volume(const int width, const int height, const int depth, const bool init = true);

    /* delete a volume */
    ~volume();

    /* init a volume */
    /* void init(const T &val); */

    /* copy a volume */
    /* volume<T> *copy() const; */

    /* get the width of a volume. */
    int width() const { return w; }

    /* get the height of a volume. */
    int height() const { return h; }

    /* get the depth of a volume. */
    int depth() const { return d; }

    /* volume data. */
    T *data;

    /* row pointers. */
    T ***access;

    inline T* operator ()(const int x, const int y) { return access[y][x]; }
    inline T& operator ()(const int x, const int y, const int z) { return access[y][x][z]; }

    /* inline T* operator ()(const int x, const int y) { return &data[y * w * d + x * d]; } */
    /* inline T& operator ()(const int x, const int y, const int z) { return data[y * w * d + x * d + z]; } */

private:
    int w, h, d;
};


template <class T>
volume<T>::volume(const int width, const int height, const int depth, const bool init) {
    w = width;
    h = height;
    d = depth;
    data = new T[w * h * d];  // allocate space for volume data
    access = new T**[h];   // allocate space for row pointers

    // initialize row pointers
    T* dataij = data;
    for (int i = 0; i < h; i++) {
        access[i] = new T*[w];
        for (int j = 0; j < w; j++, dataij += d)
            access[i][j] = dataij;
    }

    if (init)
        memset(data, 0, w * h * d * sizeof(T));
}

template <class T>
volume<T>::~volume() {
    delete [] data;
    for (int i = 0; i < h; i++)
        delete [] access[i];
    delete [] access;
}

/* template <class T> */
/* void volume<T>::init(const T &val) { */
/*     T *ptr = imPtr(this, 0, 0); */
/*     T *end = imPtr(this, w-1, h-1); */
/*     while (ptr <= end) */
/*         *ptr++ = val; */
/* } */


/* template <class T> */
/* volume<T> *volume<T>::copy() const { */
/*     volume<T> *im = new volume<T>(w, h, false); */
/*     memcpy(im->data, data, w * h * sizeof(T)); */
/*     return im; */
/* } */

#endif
