---
layout: post
title: "Python Bulit-in Time Complexity: Set & Dictionary"
category_num: 2
---

# Python Bulit-in Time Complexity: Set & Dictionary

- [Python Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Python Wiki-Time Complexity](https://wiki.python.org/moin/TimeComplexity)
- Update at : 20.12.27

## Set: Hash Table

- [cpython setobject.c](https://github.com/python/cpython/blob/master/Objects/setobject.c)
- [cpython dictobject.c](https://github.com/python/cpython/blob/master/Objects/dictobject.c)

| Opertaion | Average Case | Worst Case |
| :-------: | :----------: | :--------: |
|  search   |   $$O(1)$$   |  $$O(n)$$  |
|   copy    |   $$O(n)$$   |  $$O(n)$$  |
|    add    |   $$O(1)$$   |  $$O(n)$$  |
|  remove   |   $$O(1)$$   |  $$O(n)$$  |

Python의 Set은 **Hash Table**로 구현되어 있다. 이러한 Hash Table을 사용할 때 가장 문제되는 것은 **Hash Collision**, 즉 Hash 값에 따라 Table 상에서 저장하고자 하는 위치를 사전에 저장된 값이 이미 차지하고 있어 사용하지 못하는 경우이다. Hash Table에서 Hash Collision을 해결하기 위한 방법으로는 Chaining과 Open Address 두 가지가 대표적인데, Cpython에서는 Open Address, 그 중에서도 **Linear Probing**을 사용한다. 새로운 Element를 Set에 추가한다고 할 때 구체적인 과정은 다음과 같다.

1. Hash 값에 따라 Table 상에서 탐색을 시작할 Index를 찾는다.
2. Index부터 Max Linear Probing Size(기본값=9)만큼 Entry를 순차적으로 탐색한다.

- **동일 Element를 찾은 경우** -> Set은 중복을 허용하지 않으므로 추가하지 않는다.
- **빈 공간을 찾은 경우** -> Set 내에 동일 Element가 없다는 뜻이므로 해당 공간에 저장한다.
- **Max Linear Probing Size만큼 모두 탐색한 경우** -> PERTURB_SHIFT에 따라 다음 장소로 이동하여 다시 탐색한다.

Dynamic Array와 비교해보면 다소 복잡하지만 **Hash Table**은 위에 제시된 표에서 확인할 수 있듯이 Average Case에서의 시간 복잡도가 낮다는 장점을 가진다.

### Python Set and Dictionary

Python에서 Set과 Dictionary는 모두 Hash Table을 사용하며, Set을 구현하는 데에도 Dictionary의 구현을 많이 참조하고 있다. 가장 단순하게 말하자면 Set은 Dictionary에서 Key만 존재하는 경우라고 생각할 수 있다. 이러한 점 때문에 두 자료 구조의 시간 복잡도는 거의 동일하다.

```c
// cpython/Objects/setobject.c, set_lookkey()
if (startkey == key)
    return entry;
if (PyUnicode_CheckExact(startkey)
    && PyUnicode_CheckExact(key)
    && _PyUnicode_EQ(startkey, key))
    return entry;
```

위 코드는 Python Set에서 동일한 값이 있는지 찾는 부분이다. 첫 번째 if branch는 현재 확인하고자 하는 위치에 저장된 주소 값(startkey)과 찾고자 하는 변수의 주소 값(key)를 비교하는 부분이다. Python의 모든 요소들은 referece로 접근하게 되며, Set이나 Dictionary에도 저장된 값은 각 객체의 주소 값이 된다는 것을 여기서도 확인할 수 있다. 두 번째 if branch는 각 주소 값에 저장된 값이 동일한지 확인하는 부분이다. 둘 중 하나라도 동일한 것이라면 같은 값이 Set에 있다고 할 수 있으므로 해당 주소 값의 위치라고 할 수 있는 Entry를 반환하게 된다.

### Search

Hash Table의 약점 중 하나는 Hash Function이 어떻게 구현되는가에 따라서 자료 구조의 성능이 결정된다는 것이다. 만약 Hash Function이 항상 하나의 값만 반환한다면 Hash Table은 Static Array와 동일해진다. 따라서 Worst Case에 대한 Search 시간 복잡도는 $$O(n)$$이 된다. Average Case는 $$O(1)$$이다(`set_lookkey()`).

### Copy

모든 값을 복사해야하므로 $$O(n)$$의 시간 복잡도를 가진다.

### Add

List에서 Append를 수행하는 경우와 달리 Set에서는 같은 Hash 값을 갖는 Entry 내에 동일한 Element가 있는지 탐색하고, 없는 경우에만 새로 추가한다(`set_add_entry()`). 따라서 최악의 경우에는 Set에 저장된 모든 Element들을 확인해야 하기 때문에 Worst Case 시간 복잡도는 $$O(n)$$이 된다. 만약 Hash Table이 가득 차는 경우 여기서도 Table 자체를 재할당해준다(`set_table_resize()`).

### Remove

Search와 동일하다. 즉 Average Case는 $$O(1)$$, Worst Case는 $$O(n)$$이 된다. 참고로 Set에서는 Element를 삭제하더라도 Dynamic Array와 같이 다른 Element를 Shift 하지 않는다. 삭제된 Element에는 dummy data를 저장해두고 사용하지 않는다고 표시만 한 뒤 넘어간다(`set_discard_entry()`).

### Cpython code for Set

cpython에서 set과 관련된 코드는 다음과 같다.

```c
// cpython/Include/setobject.h
// 개별 Element가 저장되는 단위
typedef struct {
    PyObject *key;
    Py_hash_t hash;             /* Cached hash code of the key */
} setentry;

...

// 전체 Set이 저장되는 단위
// Table에는 Hash Table의 Entry가 저장된다.
// mask는 Hash Table의 전체 Slot의 개수가 저장된다.
typedef struct {
    PyObject_HEAD

    Py_ssize_t fill;            /* Number active and dummy entries*/
    Py_ssize_t used;            /* Number active entries */

    /* The table contains mask + 1 slots, and that's a power of 2.
     * We store the mask instead of the size because the mask is more
     * frequently needed.
     */
    Py_ssize_t mask;

    /* The table points to a fixed-size smalltable for small tables
     * or to additional malloc'ed memory for bigger tables.
     * The table pointer is never NULL which saves us from repeated
     * runtime null-tests.
     */
    setentry *table;
    Py_hash_t hash;             /* Only used by frozenset objects */
    Py_ssize_t finger;          /* Search finger for pop() */

    setentry smalltable[PySet_MINSIZE];
    PyObject *weakreflist;      /* List of weak references */
} PySetObject;

// cpython/Objects/setobject.c
// 특정 key 값을 가지는 Element가 있는지 확인할 때 호출
static setentry *
set_lookkey(PySetObject *so, PyObject *key, Py_hash_t hash)
{
    setentry *table;
    setentry *entry;
    size_t perturb = hash;
    size_t mask = so->mask;
    // 전체 Table에서 탐색을 시작할 Entry의 Index를 찾는다.
    size_t i = (size_t)hash & mask; /* Unsigned for defined overflow behavior */
    int probes;
    int cmp;

    while (1) {
        entry = &so->table[i];
        // Linear Probing을 고려하여 해당 Entry부터 몇 개를 확인할지 결정한다.
        probes = (i + LINEAR_PROBES <= mask) ? LINEAR_PROBES: 0;
        // probes가 0이 될 때까지 do-while을 반복한다.
        // Entry의 Hash가 찾고자하는 Hash와 다르면 do clause를 그대로 통과하여 다음 Entry를 확인한다.
        do {
            // 비어있는 Entry를 발견하면 반환한다.
            if (entry->hash == 0 && entry->key == NULL)
                return entry;
            if (entry->hash == hash) {
                PyObject *startkey = entry->key;
                assert(startkey != dummy);
                // key(주소 값)가 동일하거나
                if (startkey == key)
                    return entry;
                // key(주소 값)에 저장된 값이 동일한 경우 Entry를 반환한다.
                if (PyUnicode_CheckExact(startkey)
                    && PyUnicode_CheckExact(key)
                    && _PyUnicode_EQ(startkey, key))
                    return entry;
                table = so->table;
                Py_INCREF(startkey);
                cmp = PyObject_RichCompareBool(startkey, key, Py_EQ);
                Py_DECREF(startkey);
                if (cmp < 0)
                    return NULL;
                if (table != so->table || entry->key != startkey)
                    return set_lookkey(so, key, hash);
                if (cmp > 0)
                    return entry;
                mask = so->mask;
            }
            entry++;
        } while (probes--);
        // Max Linear probing size가 모두 가득 찬 경우 2^PERTURB_SHIFT만큼 건너뛰어 탐색한다.
        // 이때 mask를 통해 table의 전체 크기를 초과하지는 못하도록 한다.
        perturb >>= PERTURB_SHIFT;
        i = (i * 5 + 1 + perturb) & mask;
    }
}

// 새로운 Element를 추가할 때 호출
static int
set_add_entry(PySetObject *so, PyObject *key, Py_hash_t hash)
{
    setentry *table;
    setentry *entry;
    size_t perturb;
    size_t mask;
    size_t i;                       /* Unsigned for defined overflow behavior */
    int probes;
    int cmp;

    /* Pre-increment is necessary to prevent arbitrary code in the rich
       comparison from deallocating the key just before the insertion. */
    Py_INCREF(key);

  restart:

    mask = so->mask;
    i = (size_t)hash & mask;
    perturb = hash;

    while (1) {
        entry = &so->table[i];
        probes = (i + LINEAR_PROBES <= mask) ? LINEAR_PROBES: 0;
        do {
            if (entry->hash == 0 && entry->key == NULL)
                goto found_unused;
            if (entry->hash == hash) {
                PyObject *startkey = entry->key;
                assert(startkey != dummy);
                // 동일한 key가 존재하는지,
                if (startkey == key)
                    goto found_active;
                // 또는 key(주소 값)에 저장된 값이 동일한 경우가 존재하는지,
                if (PyUnicode_CheckExact(startkey)
                    && PyUnicode_CheckExact(key)
                    && _PyUnicode_EQ(startkey, key))
                    goto found_active;
                table = so->table;
                Py_INCREF(startkey);
                cmp = PyObject_RichCompareBool(startkey, key, Py_EQ);
                Py_DECREF(startkey);
                if (cmp > 0)
                    goto found_active;
                if (cmp < 0)
                    goto comparison_error;
                if (table != so->table || entry->key != startkey)
                    goto restart;
                mask = so->mask;
            }
            entry++;
        } while (probes--);
        perturb >>= PERTURB_SHIFT;
        i = (i * 5 + 1 + perturb) & mask;
    }

    found_unused:
    so->fill++;
    so->used++;
    entry->key = key;
    entry->hash = hash;
    if ((size_t)so->fill*5 < mask*3)
        return 0;
    return set_table_resize(so, so->used>50000 ? so->used*2 : so->used*4);

    // 동일한 key 또는 key에 저장된 값이 존재하는 경우
    found_active:
    Py_DECREF(key);
    return 0;

    comparison_error:
    Py_DECREF(key);
    return -1;
}

// Element를 제거할 때 호출
static int
set_discard_entry(PySetObject *so, PyObject *key, Py_hash_t hash)
{
    setentry *entry;
    PyObject *old_key;

    entry = set_lookkey(so, key, hash);
    if (entry == NULL)
        return -1;
    if (entry->key == NULL)
        return DISCARD_NOTFOUND;
    // Entry에 저장된 Element가 제거되면 key에는 dummy를, hash에는 -1을 저장한다.
    // 참고로 dummy 데이터를 저장하고 있는 경우 table_resize 시 재할당 하지 않는다.
    old_key = entry->key;
    entry->key = dummy;
    entry->hash = -1;
    so->used--;
    Py_DECREF(old_key);
    return DISCARD_FOUND;
}

/*
Restructure the table by allocating a new table and reinserting all
keys again.  When entries have been deleted, the new table may
actually be smaller than the old one.
*/
// Table을 재할당 할 때 호출
static int
set_table_resize(PySetObject *so, Py_ssize_t minused)
{
    ...
    // so->fill == so->used, 즉 dummy Entry가 없는 경우 -> dummy check를 안 한다.
    if (so->fill == so->used) {
        for (entry = oldtable; entry <= oldtable + oldmask; entry++) {
            if (entry->key != NULL) {
                set_insert_clean(newtable, newmask, entry->key, entry->hash);
            }
        }
    } else {
        so->fill = so->used;
        for (entry = oldtable; entry <= oldtable + oldmask; entry++) {
            if (entry->key != NULL && entry->key != dummy) {
                set_insert_clean(newtable, newmask, entry->key, entry->hash);
            }
        }
    }
}
```
