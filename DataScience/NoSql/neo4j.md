
## Query 1: Create a Node
```python
CREATE (p:Person {name: "John", age: 30})
```
What it does? Creates a Person node with name = John and age = 30.

How to see it?

```python
MATCH (n) RETURN n
```
## Query 2: Create Multiple Nodes
```python
CREATE (a:Person {name: "Alice", age: 25}),
       (b:Person {name: "Bob", age: 28})
```
What it does? Creates 2 people: Alice and Bob.

Check it:
```python
MATCH (n:Person) RETURN n
```
## Query 3: Read a Node with a Filter
```python
MATCH (n:Person {name: "Alice"}) RETURN n
```
What it does? Finds the Person whose name is Alice.

## Query 4: Update a Node
```python
MATCH (n:Person {name: "Alice"})
SET n.age = 26
RETURN n
```
What it does? Changes Alice's age from 25 to 26.

## Query 5: Delete a Node
```python
MATCH (n:Person {name: "Bob"})
DELETE n

```


Query 1: Create 2 Nodes First
We need at least two nodes to make a relationship.

```python
CREATE (a:Person {name: "John", age: 30}),
       (b:Person {name: "Alice", age: 26})
```
Now we have John and Alice.

Query 2: Create a Relationship
```python
MATCH (a:Person {name: "John"}), (b:Person {name: "Alice"})
CREATE (a)-[:FRIENDS_WITH]->(b)
```
What it does? Connects John to Alice with a FRIENDS_WITH relationship.

The arrow -> shows the direction (John → Alice).

Query 3: See All Relationships
```python
MATCH (x)-[r]->(y) RETURN x, r, y
```
What it does? Shows all nodes (x, y) and relationships (r).

## Query 4: Add Property to Relationship
```python
MATCH (a:Person {name: "John"})-[r:FRIENDS_WITH]->(b:Person {name: "Alice"})
SET r.since = 2020
RETURN r
```
What it does? Adds a property since: 2020 to the FRIENDS_WITH relationship.

## Query 5: Delete a Relationship
```python
MATCH (a)-[r:FRIENDS_WITH]->(b)
DELETE r
```
