//  وارد کردن کتابخانه TensorFlow.js
import * as tf from '@tensorflow/tfjs';

//  تعریف مدل
const model = tf.sequential();
model.add(tf.layers.dense({units : 1 , inputShape : [1]}));

//  تعریف و آموزش مدل
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
model.fit(xs, ys, {epochs: 10}).then(() => {
  //  پیش‌بینی مقادیر جدید
  const newXs = tf.tensor2d([5, 6], [2, 1]);
  const ysPred = model.predict(newXs);
  ysPred.print();
});
