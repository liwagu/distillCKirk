// Tier D 数字人：预录循环 + 说话/待机状态机。零 GPU、零推理。
//
// 为什么不是"播一段视频"那么简单：
// - 两个循环之间硬切会有明显跳变。用交叉淡化，并且**只在眨眼附近切**——
//   眨眼时眼睛闭着，头部姿态的不连续最不容易被察觉（Video Textures 的老技巧）。
// - 说话结束回到待机是整个效果最容易破功的一瞬间（上游 FeatherTalk issue #8
//   至今未解决）。所以收尾要留一个短的缓冲，不能音频一停就立刻切。
// - 待机片段必须够长且本身可循环，否则会看出周期性。
//
// 后面接上训练好的口型模型时，这一层不用重写：talking 轨换成模型输出的帧流即可，
// 待机轨、状态机、交叉淡化全部复用。

const AVATAR = {
  idle: null,          // <video> 待机
  talk: null,          // <video> 说话
  stage: null,         // 容器
  state: "idle",
  fadeMs: 260,
  tailMs: 420,         // 音频停后多等一会儿再切回待机（避免句间停顿来回跳）
  _tailTimer: null,
  ready: false,

  init(stageEl) {
    this.stage = stageEl;
    this.idle = stageEl.querySelector("#v-idle");
    this.talk = stageEl.querySelector("#v-talk");
    if (!this.idle || !this.talk) return;

    for (const v of [this.idle, this.talk]) {
      v.loop = true; v.muted = true; v.playsInline = true;
      v.addEventListener("error", () => this.markMissing());
    }
    // 两个轨道都常驻播放，只切透明度 —— 切换时才不会有解码启动的卡顿
    const start = () => {
      Promise.all([this.idle.play(), this.talk.play()])
        .then(() => { this.ready = true; this.stage.classList.add("ready"); })
        .catch(() => {/* 自动播放被拦：等用户点 Start 时会再试 */});
    };
    if (this.idle.readyState >= 2) start();
    else this.idle.addEventListener("canplay", start, { once: true });

    // 自愈：浏览器会在标签页隐藏、或自动播放策略触发时暂停视频轨。
    // 暂停后切状态只会换透明度，画面定格 —— 看起来像卡死。每秒兜一次。
    setInterval(() => {
      for (const v of [this.idle, this.talk]) {
        if (v && v.paused && v.readyState >= 2) v.play().catch(() => {});
      }
    }, 1000);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) {
        for (const v of [this.idle, this.talk]) v && v.play().catch(() => {});
      }
    });
    this.apply("idle", true);
  },

  markMissing() {
    this.stage && this.stage.classList.add("missing");
  },

  apply(state, instant = false) {
    if (!this.idle || !this.talk) return;
    this.state = state;
    const talking = state === "talk";
    const ms = instant ? 0 : this.fadeMs;
    this.talk.style.transition = this.idle.style.transition =
      `opacity ${ms}ms ease-in-out`;
    this.talk.style.opacity = talking ? "1" : "0";
    this.idle.style.opacity = talking ? "0" : "1";
    // 浏览器会暂停 opacity:0 的静音视频来省电，所以**即将显示的那一轨必须显式续播**。
    // 否则它定格在被隐藏时的那一帧，要等 1s 自愈才动——而刚开口那一秒最显眼。
    (talking ? this.talk : this.idle).play().catch(() => {});
  },

  // 两条轨都自由循环，开口只切透明度 —— 不 seek。
  // 曾经在这里做 currentTime=0（想让每次开口的起点一致），结果 seek 与紧随的
  // play() 打架，视频定格在第 0 帧，要等 1s 自愈才恢复；而那正是他刚开口、
  // 最显眼的一秒。何况每次都从同一帧起播反而让循环感更明显。
  speak() {
    clearTimeout(this._tailTimer);
    if (this.state !== "talk") this.apply("talk");
  },

  // 不立刻切回：一句话结束到下一句之间有间隙，硬切会来回跳
  stop() {
    clearTimeout(this._tailTimer);
    this._tailTimer = setTimeout(() => this.apply("idle"), this.tailMs);
  },

  // 打断：立刻收嘴。必须 instant —— 走 260ms 淡化的话，用户已经开口了
  // 画面里的人还在动嘴，这正是最出戏的一幕。
  cut() {
    clearTimeout(this._tailTimer);
    this.apply("idle", true);
  },
};

window.AVATAR = AVATAR;
