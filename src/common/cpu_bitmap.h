/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include "gl_helper.h"
#include "free_image/FreeImage.h"
#include "stdio.h"

struct CPUBitmap {
    unsigned char    *pixels;
    int     x, y;
    void    *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height, void *d = NULL ) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return x * y * 4; }

    void display_and_exit( void(*e)(void*) = NULL) {
        CPUBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        bitmapExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
        glutInitWindowSize( x, y );
        glutCreateWindow( "bitmap" );
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        glutMainLoop();
    }

    void save_to_file(const char *file_name) {
        CPUBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        bitmapExit = nullptr;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c = 1;
        char* dummy = "";
        glutInit(&c, &dummy);
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
        glutInitWindowSize(x, y);
        glutCreateWindow("bitmap");
        glutKeyboardFunc(Key);
        DrawToFile(file_name);
    }

     // static method used for glut callbacks
    static CPUBitmap** get_bitmap_ptr( void ) {
        static CPUBitmap   *gBitmap;
        return &gBitmap;
    }

   // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                CPUBitmap*   bitmap = *(get_bitmap_ptr());
                if (bitmap->dataBlock != NULL && bitmap->bitmapExit != NULL)
                    bitmap->bitmapExit( bitmap->dataBlock );
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw() {
        CPUBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
        glFlush();
    }

    // static method used for glut callbacks
    static void DrawToFile(const char *file_name) {
        CPUBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->x, bitmap->y, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels);

        if (file_name != nullptr) {
            // Make the BYTE array, factor of 3 because it's RBG.

            printf("Saving image...\n");
            BYTE* pixels = new BYTE[3 * bitmap->x * bitmap->y];

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glReadPixels(0, 0, bitmap->x, bitmap->y, GL_RGB, GL_UNSIGNED_BYTE, pixels);

            // Convert to FreeImage format & save to file
            FIBITMAP* image = FreeImage_ConvertFromRawBits(
                pixels, bitmap->x, bitmap->y, 3 * bitmap->x, 24, 0xFF0000, 0x00FF00, 0x0000FF, false
            );
            FreeImage_Save(FIF_PNG, image, file_name, 0);

            // Free resources
            FreeImage_Unload(image);
            delete[] pixels;
            printf("Saved image.\n");
        }

        glFlush();
    }
};

#endif  // __CPU_BITMAP_H__
