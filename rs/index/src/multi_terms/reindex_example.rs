//! Example demonstrating how to use the reindex functionality for MultiTerms
//!
//! This example shows how to:
//! 1. Create a MultiTermBuilder with multiple users
//! 2. Generate ID mappings (simulating what would come from SPANN reindex)
//! 3. Write the multiterm index with reindexing applied

use std::collections::HashMap;

use crate::multi_terms::builder::MultiTermBuilder;
use crate::multi_terms::index::MultiTermIndex;
use crate::multi_terms::writer::MultiTermWriter;

pub fn example_multiterm_reindex() -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary directory for the example
    let temp_dir = tempdir::TempDir::new("multiterm_reindex_example")?;
    let base_dir = temp_dir.path().to_str().unwrap().to_string();

    // Create a MultiTermBuilder and add terms for multiple users
    let mut builder = MultiTermBuilder::new();

    // User 1: Document IDs 0, 1, 2, 3, 4
    let user1 = 100u128;
    builder.add(user1, 0, "apple".to_string())?;
    builder.add(user1, 1, "banana".to_string())?;
    builder.add(user1, 2, "apple".to_string())?;
    builder.add(user1, 3, "cherry".to_string())?;
    builder.add(user1, 4, "banana".to_string())?;

    // User 2: Document IDs 0, 1, 2
    let user2 = 200u128;
    builder.add(user2, 0, "dog".to_string())?;
    builder.add(user2, 1, "cat".to_string())?;
    builder.add(user2, 2, "dog".to_string())?;

    // Build the term builder
    builder.build()?;

    // Simulate ID mappings from SPANN reindex
    // User 1: Original IDs [0,1,2,3,4] -> New IDs [2,0,4,1,3] (reordered)
    // User 2: Original IDs [0,1,2] -> New IDs [0,2,1] (reordered)
    let mut id_mappings = HashMap::new();
    id_mappings.insert(user1, vec![2, 0, 4, 1, 3]);
    id_mappings.insert(user2, vec![0, 2, 1]);

    // Write the multiterm index with reindexing applied
    let writer = MultiTermWriter::new(base_dir.clone());
    writer.write_with_reindex(&mut builder, Some(&id_mappings))?;

    // Load the index and verify the reindexing worked
    let index = MultiTermIndex::new(base_dir)?;

    // Verify User 1's terms were remapped correctly
    println!("User 1 terms after reindex:");

    // "apple" should now contain points [2, 4] (original points 1, 2)
    // These will be sorted to [2, 4]
    let apple_id = index.get_term_id_for_user(user1, "apple")?;
    let apple_pl: Vec<u32> = index
        .get_or_create_index(user1)?
        .get_posting_list_iterator(apple_id)?
        .collect();
    println!("  apple: {:?}", apple_pl);
    assert_eq!(apple_pl, vec![2, 4]);

    // "banana" should now contain points [0, 3] (original points 0, 4)
    // These will be sorted to [0, 3]
    let banana_id = index.get_term_id_for_user(user1, "banana")?;
    let banana_pl: Vec<u32> = index
        .get_or_create_index(user1)?
        .get_posting_list_iterator(banana_id)?
        .collect();
    println!("  banana: {:?}", banana_pl);
    assert_eq!(banana_pl, vec![0, 3]);

    // "cherry" should now contain point [1] (original point 3)
    let cherry_id = index.get_term_id_for_user(user1, "cherry")?;
    let cherry_pl: Vec<u32> = index
        .get_or_create_index(user1)?
        .get_posting_list_iterator(cherry_id)?
        .collect();
    println!("  cherry: {:?}", cherry_pl);
    assert_eq!(cherry_pl, vec![1]);

    // Verify User 2's terms were remapped correctly
    println!("User 2 terms after reindex:");

    // "dog" should now contain points [0, 1] (original points 0, 2)
    let dog_id = index.get_term_id_for_user(user2, "dog")?;
    let dog_pl: Vec<u32> = index
        .get_or_create_index(user2)?
        .get_posting_list_iterator(dog_id)?
        .collect();
    println!("  dog: {:?}", dog_pl);
    assert_eq!(dog_pl, vec![0, 1]);

    // "cat" should now contain point [2] (original point 1)
    let cat_id = index.get_term_id_for_user(user2, "cat")?;
    let cat_pl: Vec<u32> = index
        .get_or_create_index(user2)?
        .get_posting_list_iterator(cat_id)?
        .collect();
    println!("  cat: {:?}", cat_pl);
    assert_eq!(cat_pl, vec![2]);

    println!("Reindex example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_multiterm_reindex() {
        example_multiterm_reindex().unwrap();
    }
}
