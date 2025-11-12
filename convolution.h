#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "design-space.h"

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 3
#endif

#ifndef STRIDE
#define STRIDE 1
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 4
#endif

#ifndef TILE_HEIGHT
#define TILE_HEIGHT 7
#endif

#ifndef OVERLAP
#define OVERLAP (KERNEL_SIZE - STRIDE)
#endif

#ifndef STEP_X
#define STEP_X (TILE_HEIGHT - OVERLAP)
#endif

#ifndef STEP_Y
#define STEP_Y (TILE_WIDTH - OVERLAP)
#endif

#ifndef TILES_X
#define TILES_X ((ROWS + STEP_X - 1) / STEP_X)
#endif

#ifndef TILES_Y
#define TILES_Y ((COLS + STEP_Y - 1) / STEP_Y)
#endif

#ifndef TILEINFO_STRUCT
#define TILEINFO_STRUCT
struct TileInfo { // this is for easy statistics only
  // only the active field is necessary
  bool active;
  int non_empty;
  int empty;
};
#endif

#endif // CONVOLUTION_H
