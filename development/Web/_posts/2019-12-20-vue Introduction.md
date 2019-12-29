---
layout: post
title: Vue Introduction
---

# Vue Introduction

## Vue installation and running

추가하기

## Vue component의 구조

Vue component는 기본적으로 template, script, style 세 부분으로 구성되어 있다. template는 component의 내용을 구성하는 부분, 즉 html과 유사한 역할을 한다. 문법 또한 html과 매우 닮아있다.

```
 <template>
  <div id="app">
    <Todos />
  </div>
</template>
```

script는 js 문법을 사용하며, 메타 정보들을 처리하는 공간이다. script tag 내에서는 component에서 사용할 다른 component을 import하거나 name, components, methods, data 등과 같은 다양한 parameter를 선언하여 component의 export name, component에서 사용할 components, data, event 처리 등을 정의한다.

```
<script>
import Todos from './components/Todos';

export default {
  name: 'app',
  components: {
    Todos
  },
  data() {
    return {
      todos: [
        {
          id : 1,
          title : "Todo One",
          completed : false
        },
        {
          id : 2,
          title : "Todo Two",
          completed : true
        },
        {
          id : 3,
          title : "Todo Three",
          completed : false
        }
      ]
    }
  },
  methods : {

  }
}
</script>
```

style tag는 css와 동일하게 작성하면 된다.

## Vue script tag의 이해

script tag의 export default {} 내의 여러 property로 component를 제어할 수 있다.

### components property

component에서 사용할 child component를 정의할 때 사용한다. import 한 component를 components property로 등록해주어야 사용이 가능하다.

### data property

기본적인 설정값을 전달할 수 있다. data property는 객체로 전달할 수도 있지만 함수를 이용해 값을 return 하도록 하는 방법이 더 좋다. 이렇게 하면 Vue directive로도 사용이 가능해지기 때문이다.

또 하나 주의해야 할 점으로는 arrow function을 사용하면 안된다는 점이다. this binding이 되지 않기 때문이다.

### methods property

event 등이 발생했을 때 실행되는 함수를 정의하는 property이다. methods에서 addTodo라는 함수를 정의하면 v-on directive를 통해 사용할 수 있다.

여기서도 arrow function을 사용하면 안된다.

## Vue directive

Vue에서 제공하는 다양한 기능들을 사용하기 위한 대표적인 방법으로, "v-" 접두사를 가지고 있다. html tag 내에 하나의 속성값 형태로 사용된다.

directive에 할당하는 값은 기본적으로 모두 string이다.

### v-bind

template tag 내의 html element 를 설정할 때 {{ image }} 와 같이 script 내의 값을 바로 받으려고 하면 동작하지 않는다. 이러한 문제를 해결하기 위해 사용하는 것이 v-bind directive이다. 한마디로 js로 선언된 값을 html이 알아들을 수 있도록 하는 것이다.

- 원래 html img tag 사용 방법

    ```
    <img src="링크"/>
    ```

- 위 방법을 그대로 .vue 파일에서 사용하면 error가 발생한다.

    ```
    <img src={{ image }}/>
    <img src"{{ image }}"/>
    ```

- v-bind를 사용하면 된다.

    ```
    <img v-bind:src="image"/>
    ```

- v-vind는 : 으로 축약되기도 한다.

    ```
    <img :src="image"/>
    ```

### v-for directive

특정 tag를 반복적으로 생성할 때 사용한다. v-for="<element> in <elements>"(여기서 elements는 iterative object여야 한다) 와 같이 사용한다.

    ```
    <div v-bind:key="todo.id" v-for="todo in todos">
        <TodoItem v-bind:todo="todo" v-on:del-todo='$emit("del-todo", todo.id)' />
    </div>
    ```

위와 같이 하면 div tag가 todos의 element 개수만큼 만들어진다.

### v-if directive

쉽게 말해 if 값이 true 일 때에만 해당 tag가 보이게 할 때 사용하는 directive이다.

    ```
    <div id="app">
        <h1 v-if="value > 5">value 가 5보다 크군요</h1>
    </div>
    ```

v-else 등의 directive도 동일한 방식으로 사용할 수 있다.

### v-on directive

event를 다루기 위해 사용하는 directive이다. 기본적인 사용방법은 다음과 같다.

    ```
    v-on:<event_name>="<method_name>"
    ```

이 경우 event가 감지되면 methods에 등록된 함수가 동작한다.

    ```
    @<event_name>="<method_name>"
    ```

와 같이 @로 대체할 수도 있다.

### v-model directive

기본적으로 template는 script에서 data property로 전달된 값들을 사용하도록 되어 있다. 이와는 반대로 template에서 data property를 바꾸고 싶을 때, 즉 양방향 바인딩을 가능하도록 해주는 것이 v-model directive이다.

v-model:"<data_name>"

이 경우 data_name과 해당 html tag가 연결된다. 아래는 v-model directive의 대표적인 예시로, data property 중 title의 값이 text input value에 따라 달라지게 된다.

    ```
    <input type="text" v-model="title" name = "title" placeholder="Add Todo...">
    ```

## Component 간 정보 전달

### child component로 정보 전달하기

1. (parent component) template에서 html property로 전달하고자 하는 data property 전달하기

    ```
    <TodoItem v-bind:todo="todo" v-on:del-todo='$emit("del-todo", todo.id)' />
    ```

    이 경우 todo 라는 이름으로 todo object를 전달한다.

2. (child component) script에서 props property로 전달된 데이터를 받아들인다.

    ```
    <script>
        export default {
            name : "TodoItem",
            props : ["todo"],
            methods: {
                markComplete() {
                    this.todo.completed = !this.todo.completed
                }
            }
        }
    </script>
    ```

    props property는 list 내에 string으로 선언하며, 이렇게 하면 todo를 사용할 수 있게 된다.

3. (child component) template에서 props 값을 사용한다. data property로 선언한 값과 크게 다르지 않다.

### parent component로 정보 전달하기

parent component로 데이터를 보내는 가장 쉬운 방법은 event interface를 사용하는 방법이다. event interface로는 $on, $emit 두 가지가 있으며, 각각 event listener, event trigger 라고 불린다.

하나의 component 내에서는 $emit를 통해 발생한 event를 $on 이 감지할 수 있다. 하지만 다른 component에서는 이를 감지하지 못한다. 이러한 경우에는 v-on directive를 사용해야 한다.

    ```
    <button @click="$emit('del-todo', todo.id)" class="del">x</button>
    ```

child component(TodoItem)에서 button을 클릭하면 'del-todo'라는 이름의 event가 발생하게 되며, 이 때 todo.id object를 함께 전달하게 된다.

    ```
    <TodoItem v-bind:todo="todo" v-on:del-todo='$emit("del-todo", todo.id)' />
    ```

parent component(Todo)에서는 v-on derective로 event를 감지하면 된다. 여기서는 del-todo event를 감지하면 다시 del-todo event를 발생시키도록 되어 있다.

## style

### scoped

style tag를 선언할 때 scoped를 함께 전달하면 해당 component 내에서만 style이 적용된다.

```
<style scoped>
</style>
```

## Vue LifeCycle

Vue에서 life cycle은 크게 Creation, Mounting, Updating, Destruction 네 단계로 나누어진다. 각각의 단계에서 호출되는 훅(Hook)은 다음과 같으며, script tag 내에서 export property 로 정의되는 방식으로 동작한다.

- Creation : beforeCreate, created
- Mounting : beforeMount, mounted
- Updating : beforeUpdated, updated
- Destructing : beforeDestroy, destroyed

[링크](<https://medium.com/witinweb/vue-js-%EB%9D%BC%EC%9D%B4%ED%94%84%EC%82%AC%EC%9D%B4%ED%81%B4-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-7780cdd97dd4>)
