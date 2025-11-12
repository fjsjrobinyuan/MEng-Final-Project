#include "convolution.h"
#include "design-space.h"

void generate_random_input_space(int (&input_space)[ROWS][COLS]) {
  for (int i = 0; i < ROWS; ++i) {
    for (int j = 0; j < COLS; ++j) {
      if (i >= NON_EMPTY_TOP_LEFT_X && j >= NON_EMPTY_TOP_LEFT_Y &&
          i <= NON_EMPTY_BOTTOM_RIGHT_X && j <= NON_EMPTY_BOTTOM_RIGHT_Y) {
        input_space[i][j] = rand() % 100 + 1; // non-zero
      } else {
        input_space[i][j] = 0;
      }
    }
  }
}

int main() {
  std::cout << "Tile selection module" << std::endl;

  int input_space[ROWS][COLS] = {0};
  generate_random_input_space(input_space);

  std::vector<std::vector<TileInfo>> tiles(
      TILES_X, std::vector<TileInfo>(TILES_Y, {false, 0, 0}));

  for (int tx = 0; tx < TILES_X; ++tx) {
    for (int ty = 0; ty < TILES_Y; ++ty) {

      int start_x = tx * STEP_X;
      int start_y = ty * STEP_Y;
      int end_x = std::min(start_x + TILE_HEIGHT, ROWS);
      int end_y = std::min(start_y + TILE_WIDTH, COLS);

      int non_empty = 0;
      int empty = 0;

      for (int i = start_x; i < end_x; ++i) {
        for (int j = start_y; j < end_y; ++j) {
          if (input_space[i][j] != 0)
            ++non_empty;
          else
            ++empty;
        }
      }

      tiles[tx][ty].non_empty = non_empty;
      tiles[tx][ty].empty = empty;
      tiles[tx][ty].active = (non_empty > 0);
    }
  }

  std::cout << "\nTile statistics (active tiles and pixel counts):\n";
  for (int tx = 0; tx < TILES_X; ++tx) {
    for (int ty = 0; ty < TILES_Y; ++ty) {
      const auto &t = tiles[tx][ty];
      if (t.active) {
        std::cout << "Tile (" << tx << ", " << ty << ") "
                  << "non-empty: " << t.non_empty << ", empty: " << t.empty
                  << std::endl;
      }
    }
  }

  return 0;
}
