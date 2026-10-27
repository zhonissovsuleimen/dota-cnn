use std::fs;

use regex::Regex;

#[derive(Debug)]
struct Hero {
  name: String,
  _type: String,
  math: String,
}

fn main() {
  let file = fs::read_to_string("sets.txt").expect("No stats file found");

  let regex = Regex::new(
    r"(?m)^\r?\n(?<name>[^\r\n()]+?)(?:[ \t]+\((?<type>[^()\r\n]*)\))?\r?\n(?<math>[^\r\n]+)$",
  )
  .unwrap();

  let heroes = regex.captures_iter(&file).map(|caps| Hero {
    name: caps["name"].to_owned(),
    _type: caps.name("type").map_or("", |m| m.as_str()).to_owned(),
    math: caps["math"].to_owned(),
  });

  let mut hm = std::collections::HashMap::<String, (u64, u64)>::new();

  for hero in heroes {
    let totals = hero.math.split('*').fold((1u64, 1u64), |totals, math| {
      let number: u64 = math
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .parse()
        .expect(&format!("Error parsing string: {}", math));

      (totals.0 * number, totals.1 * (number + 1))
    });

    hm.entry(hero.name)
      .and_modify(|n| *n = (n.0 + totals.0, n.1 + totals.1))
      .or_insert(totals);
  }

  let totals = hm.values().fold((0u64, 0u64), |accum, curr| {
    (accum.0 + curr.0, accum.1 + curr.1)
  });

  println!("Total permuatitions: {:.2e}", totals.0);
  println!("Total permuatitions with default skins: {:.2e}", totals.1);

  let mut sorted: Vec<(String, (u64, u64))> = hm.iter().map(|(k, &v)| (k.clone(), v)).collect();
  sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));

  println!("Heros with most permuatitions: ");
  for i in 0..3 {
    println!("{} ({:.2e})", sorted[i].0, sorted[i].1.1);
  }
}
