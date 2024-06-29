use std::time::Instant;
use std::f64::NEG_INFINITY;

// Struct representing each cell in the DP table with its value and parent coordinates
#[derive(Clone, Copy)]
struct Cell {
    value: f64,
    parent: Option<(usize, usize)>,
}

// Struct for the dynamic programming path finding algorithm
struct DynamicProgramming {
    grid: Vec<Vec<f64>>,
    start: (usize, usize),
    t_max: usize,
    regen_value: f64,
    rows: usize,
    cols: usize,
    dp: Vec<Vec<Vec<Cell>>>,
    directions: Vec<(isize, isize)>,
}

impl DynamicProgramming {
    // Constructor to initialize the dynamic programming structure
    fn new(grid: Vec<Vec<f64>>, start: (usize, usize), t_max: usize, regen_value: f64) -> Self {
        let rows = grid.len();
        let cols = grid[0].len();
        let dp = vec![
            vec![
                vec![
                    Cell {
                        value: NEG_INFINITY,
                        parent: None,
                    };
                    t_max + 1
                ];
                cols
            ];
            rows
        ];
        let directions = vec![
            (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1)
        ];

        Self {
            grid,
            start,
            t_max,
            regen_value,
            rows,
            cols,
            dp,
            directions,
        }
    }

    // Method to reconstruct the path from a given cell and time back to the start
    fn reconstruct_path(&self, mut x: usize, mut y: usize, mut t: usize) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        while t > 0 {
            path.push((x, y));
            if let Some((px, py)) = self.dp[x][y][t].parent {
                x = px;
                y = py;
            }
            t -= 1;
        }
        path.push((self.start.0, self.start.1));
        path.reverse();
        path
    }

    // Main method to calculate the maximum path value and the corresponding path
    fn max_path_value_and_path(&mut self) -> (f64, Vec<(usize, usize)>, Vec<Vec<Vec<Cell>>>) {
        // Initialize the starting position with the initial grid value
        self.dp[self.start.0][self.start.1][0].value = self.grid[self.start.0][self.start.1];

        let start_time = Instant::now();
        
        // Iterate over each time step
        for t in 1..=self.t_max {
            // Iterate over the subset of the grid within the T/2 distance from the start
            for x in self.start.0.saturating_sub(self.t_max / 2)..=(self.start.0 + self.t_max / 2).min(self.rows - 1) {
                for y in self.start.1.saturating_sub(self.t_max / 2)..=(self.start.1 + self.t_max / 2).min(self.cols - 1) {
                    let current_value = self.dp[x][y][t - 1].value;
                    // Consider all possible movements
                    for &(dx, dy) in &self.directions {
                        let nx = x.wrapping_add(dx as usize);
                        let ny = y.wrapping_add(dy as usize);
                        if nx < self.rows && ny < self.cols {
                            // Reconstruct the path to check visited cells
                            let visited_cells = self.reconstruct_path(x, y, t - 1);
                            let new_value = if visited_cells.contains(&(nx, ny)) {
                                // Apply regeneration if cell was already visited
                                current_value + (self.grid[nx][ny]).min((visited_cells.iter().position(|&(vx, vy)| vx == nx && vy == ny).unwrap() as f64) * self.regen_value)
                            } else {
                                current_value + self.grid[nx][ny]
                            };
                            // Update dp table and parent table if a better value is found
                            if new_value > self.dp[nx][ny][t].value {
                                self.dp[nx][ny][t].value = new_value;
                                self.dp[nx][ny][t].parent = Some((x, y));
                            }
                        }
                    }
                }
            }
        }

        let elapsed_time = start_time.elapsed();
        let elapsed_time_ms = elapsed_time.as_secs_f64() * 1000.0;

        // Get the maximum value and the path from the start position
        let max_value = self.dp[self.start.0][self.start.1][self.t_max].value;
        let path = self.reconstruct_path(self.start.0, self.start.1, self.t_max);

        (max_value, path, self.dp.clone())
    }
}

fn main() {
    // Example grid
    let grid = vec![
        vec![0.0, 1.0, 2.0],
        vec![3.0, 4.0, 5.0],
        vec![6.0, 7.0, 8.0]
    ];
    let start = (0, 0);
    let t_max = 5;
    let regen_value = 0.5;

    // Initialize and run the dynamic programming path finder
    let mut dp = DynamicProgramming::new(grid, start, t_max, regen_value);
    let (max_value, path, _dp) = dp.max_path_value_and_path();

    // Print the results
    println!("Max value: {}", max_value);
    println!("Path: {:?}", path);
}
