# -*- coding: utf-8 -*-
"""MP_OPT learning lab. Educational implementation, NOT a reproduction of published experiments."""
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MP-OPT Interactive Research Lab", page_icon="🔬", layout="wide")
st.markdown("""<style>
  .block-container {padding-top:1.5rem;max-width:1360px}
  div[data-testid='stMetricValue'] {font-size:1.5rem}
  div[data-testid='stAlert'] {border-radius:12px}
  div[data-testid='stVerticalBlockBorderWrapper'] {border-radius:14px}
  h1,h2,h3 {letter-spacing:-.02em}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def create_series(days, points_per_day, day, offset, duration, strength, noise, seed):
    rng=np.random.default_rng(seed)
    n=days*points_per_day
    phase=(np.arange(n)%points_per_day)/points_per_day
    base=20+8*np.sin(2*np.pi*(phase-.22))+3*np.sin(4*np.pi*(phase-.15))
    y=base+rng.normal(0,noise,n)
    a=(day-1)*points_per_day+offset
    b=min(a+duration,n)
    if b>a:
        # Unusual locally fluctuating shape; adjustable amplitude, not a guaranteed max score.
        anomalous_wave=np.sin(np.linspace(0,2.5*np.pi,b-a))*0.7-0.4
        y[a:b]+=strength*anomalous_wave
    return y,base,(a,b)

def normalized(v):
    v=np.asarray(v,dtype=float); sigma=float(np.std(v))
    if sigma<1e-10:return np.zeros_like(v)
    return (v-np.mean(v))/sigma

def distance(x,y,length_adjust=True):
    z=normalized(x)-normalized(y)
    return float(np.linalg.norm(z)/(math.sqrt(len(z)) if length_adjust else 1))

@st.cache_data(show_spinner=False)
def stomp_profile(tup,m):
    """Univariate STOMP diagonal dot-product recurrence; population std z-distance."""
    x=np.asarray(tup,dtype=float); n=len(x); N=n-m+1
    if N<2:return np.full(max(0,N),np.nan),np.full(max(0,N),-1)
    means=np.array([np.mean(x[i:i+m]) for i in range(N)])
    std=np.array([np.std(x[i:i+m]) for i in range(N)])
    first=np.array([np.dot(x[:m],x[j:j+m]) for j in range(N)])
    dots=first.copy(); profile=np.full(N,np.inf); neighbors=np.full(N,-1,dtype=int)
    exclusion=max(1,math.ceil(m/4))
    for i in range(N):
        if i:
            old=dots.copy()
            dots[1:]=old[:-1]-x[i-1]*x[:N-1]+x[i+m-1]*x[m:m+N-1]
            dots[0]=first[i] # symmetry: dot(T_i,T_0)
        den=m*std[i]*std
        corr=np.divide(dots-m*means[i]*means,den,out=np.zeros(N),where=den>1e-10)
        ds=np.sqrt(np.maximum(0,2*m*(1-np.clip(corr,-1,1))))
        # exact normalized fallback for constant windows
        if std[i]<1e-10 or np.any(std<1e-10):
            const_idxs=np.where(std<1e-10)[0]
            if std[i]<1e-10:
                ds=np.array([distance(x[i:i+m],x[j:j+m],False) for j in range(N)])
            else:
                ds[const_idxs]=np.array([distance(x[i:i+m],x[j:j+m],False) for j in const_idxs])
        ds[np.abs(np.arange(N)-i)<=exclusion]=np.inf
        j=int(np.argmin(ds))
        if np.isfinite(ds[j]):profile[i]=ds[j];neighbors[i]=j
    return profile,neighbors

def topk_indices(profile,k,m):
    result=[]
    for i in np.argsort(-np.nan_to_num(profile,nan=-np.inf)):
        if not np.isfinite(profile[i]):continue
        # Prevent nearly identical overlapping top-k detections.
        if all(abs(int(i)-j)>=m for j in result):result.append(int(i))
        if len(result)>=k:break
    return result

def lhs_positions(n,anchor,radius,q,base_window,seed):
    """1D Latin hypercube-style stratification over a bounded temporal neighborhood."""
    lo=max(0,anchor-radius);hi=min(n-base_window,anchor+radius)
    if hi<lo:return []
    rng=np.random.default_rng(seed+119+anchor)
    # each stratum yields one candidate, independently shuffled for reproducibility
    edges=np.linspace(lo,hi+1,q+1)
    pts=[]
    for j in rng.permutation(q):
        left=math.ceil(edges[j]);right=max(left,math.floor(edges[j+1]-1))
        pts.append(int(rng.integers(left,right+1)))
    return sorted(set(pts))

def score_candidate(x,i,m,refs):
    i=int(round(i));m=int(round(m))
    if m<3 or i<0 or i+m>len(x):return -1e9,0
    usable=[j for j in refs if j>=0 and j+m<=len(x) and (j+m<=i or i+m<=j)]
    if not usable:return -1e9,0
    return float(np.mean([distance(x[i:i+m],x[j:j+m]) for j in usable])),len(usable)

def objective(x,i,m,refs):
    s,_=score_candidate(x,i,m,refs)
    return -s if s>-1e8 else 1e9

def clip_round(vec, bounds):
    return np.array([int(np.clip(round(vec[j]),bounds[j][0],bounds[j][1])) for j in range(2)],dtype=float)

def discover_groups(pop):
    """Use genuine HDBSCAN if installed; otherwise explicit single-group educational fallback."""
    try:
        from sklearn.cluster import HDBSCAN
        scaled=(pop-pop.mean(axis=0))/np.maximum(pop.std(axis=0),1.0)
        labels=HDBSCAN(min_cluster_size=2,min_samples=2).fit_predict(scaled)
        method="HDBSCAN（scikit-learn）"
    except (ImportError, AttributeError):
        labels=np.zeros(len(pop),dtype=int)
        method="單群簡化模式（尚未安裝 scikit-learn，非 HDBSCAN）"
    if not any(np.sum(labels==v)>=2 for v in set(labels) if v>=0):
        labels=np.zeros(len(pop),dtype=int)
        method+="；本次未形成有效群，退回單群"
    return np.asarray(labels),method

@st.cache_data(show_spinner=False)
def run_csos(series,refs_in,bounds_in,seed,popsize,maxfe):
    """CSOS-inspired working implementation: mutualism, commensalism, parasitism,
    static initial HDBSCAN groups, bounded integer solutions; suitable for teaching.
    Paper reproduction would require author settings, full code, and validation."""
    x=np.array(series);refs=list(refs_in); bounds=[list(a) for a in bounds_in]
    rng=np.random.default_rng(seed+991)
    pop=np.array([[rng.integers(b[0],b[1]+1) for b in bounds] for _ in range(popsize)],float)
    groups,group_method=discover_groups(pop)
    cache={}
    evals=0
    def fit(v):
        nonlocal evals
        key=tuple(clip_round(v,bounds).astype(int))
        if key not in cache:
            cache[key]=objective(x,*key,refs);evals+=1
        return cache[key]
    fitness=np.array([fit(v) for v in pop]);history=[float(fitness.min())]
    snapshots=[pop.copy()];maxloops=max(1,min(100,maxfe//max(popsize,1)))
    for it in range(maxloops):
        if evals>=maxfe:break
        valid=[g for g in np.unique(groups) if g>=0 and np.sum(groups==g)>=2]
        local_best={int(g):pop[np.where(groups==g)[0][np.argmin(fitness[groups==g])]].copy() for g in valid}
        for a in range(popsize):
            if evals>=maxfe:break
            g=groups[a]
            peers=np.where(groups==g)[0] if g>=0 else np.array([],dtype=int)
            peers=peers[peers!=a]
            if len(peers)==0:continue
            b=int(rng.choice(peers));LB=local_best[int(g)]
            # mutualism: update both individuals if improvement
            mv=(pop[a]+pop[b])/2
            for idx,bf in ((a,int(rng.integers(1,3))),(b,int(rng.integers(1,3)))):
                trial=clip_round(pop[idx]+rng.random(2)*(LB-bf*mv),bounds)
                f=fit(trial)
                if f<fitness[idx]:pop[idx],fitness[idx]=trial,f
            # commensalism: only one individual changes
            b=int(rng.choice(peers))
            trial=clip_round(pop[a]+rng.uniform(-1,1,2)*(LB-pop[b]),bounds)
            f=fit(trial)
            if f<fitness[a]:pop[a],fitness[a]=trial,f
        # parasitism: mutate a local best and challenge a different cluster best.
        if len(valid)>=2 and evals<maxfe:
            g,h=rng.choice(valid,2,replace=False)
            idx_g=np.where(groups==g)[0];idx_h=np.where(groups==h)[0]
            target=idx_g[np.argmin(fitness[idx_g])]; donor=idx_h[np.argmin(fitness[idx_h])]
            trial=pop[donor].copy();dimension=int(rng.integers(0,2))
            trial[dimension]=rng.integers(bounds[dimension][0],bounds[dimension][1]+1)
            trial=clip_round(trial,bounds);f=fit(trial)
            if f<fitness[target]:pop[target],fitness[target]=trial,f
        history.append(float(fitness.min()))
        if it%2==0:snapshots.append(pop.copy())
        if len(history)>12 and np.allclose(history[-10:],history[-1]):break
    best=pop[np.argmin(fitness)].astype(int)
    return tuple(best.tolist()),float(-np.min(fitness)),history,groups,group_method,snapshots,evals

def show_series(y,ab,selection=None,stage_start=None,downstream=None,title="時間序列"):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y)),y=y,name="車流量",line=dict(color="#396486",width=2)))
    fig.add_vrect(x0=ab[0]-.5,x1=ab[1]-.5,fillcolor="crimson",opacity=.16,line_width=0,annotation_text="植入的異常",annotation_position="top left")
    if selection:
        i,m=selection
        fig.add_vrect(x0=i-.5,x1=i+m-.5,fillcolor="seagreen",opacity=.22,line_width=0,annotation_text="候選片段",annotation_position="top right")
    if stage_start is not None:fig.add_vline(x=stage_start,line_dash="dash",line_color="#4b81cc")
    fig.update_layout(title=title,height=340,margin=dict(l=12,r=12,t=42,b=12),xaxis_title="資料索引",yaxis_title="車流量",legend=dict(orientation="h"))
    st.plotly_chart(fig,width="stretch")

def dtw(a,b):
    a=normalized(a);b=normalized(b)
    dp=np.full((len(a)+1,len(b)+1),np.inf);dp[0,0]=0
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i,j]=abs(a[i-1]-b[j-1])+min(dp[i-1,j],dp[i,j-1],dp[i-1,j-1])
    return float(dp[-1,-1]/(len(a)+len(b)))

st.title("🔬 MP-OPT｜互動式研究實驗室")
st.caption("Two-Stage Variable-Length Anomaly Detection · Sutrisno & Phoa (2025)")
st.info("這是基於論文概念設計的互動式教學模擬，不是論文作者的官方程式，也不是正式實驗重現。資料均為人工產生，可安全用於公開展示。")
with st.sidebar:
    st.header("⚙️ 先設定實驗資料")
    seed=st.number_input("隨機種子（讓結果重現）",1,999,12,1)
    days=st.slider("總天數",8,22,12)
    ppd=st.select_slider("每一天幾筆資料",options=[8,12,16],value=12)
    day=st.slider("哪一天發生異常",2,days-1,min(6,days-1))
    offset=st.slider("當天從第幾筆開始（0 起算）",0,ppd-2,min(6,ppd-2))
    duration=st.slider("異常持續幾筆",2,min(12,days*ppd-((day-1)*ppd+offset)),5)
    strength=st.slider("異常強度",0.,25.,15.,1.)
    noise=st.slider("正常雜訊",0.,2.,.5,.1)
    st.info("左側控制的是『人工製造的真實異常』；演算法不會直接讀取真實答案。")

y,base,ab=create_series(days,ppd,day,offset,duration,strength,noise,seed)
n=len(y)
tabs=st.tabs(["🧭 0. 導覽", "🔎 1. MP + STOMP", "🎯 2. LHS + 分數", "🤖 3. HDBSCAN + CSOS", "🛣️ 4. 跨測站", "📊 5. 論文結果與總結"])
with tabs[0]:
    st.subheader("看懂整篇論文，只要走完這 5 關")
    st.markdown("""**研究問題：** 在長時間車流量中，異常不知道從哪裡開始、持續多長。\n\n**Stage 1（MP + STOMP）** 先在整份資料中找出 Top-k 個可疑的固定長度片段。\n\n**Stage 2（LHS + HDBSCAN + CSOS）** 分別在候選附近調整 `(起點 i, 長度 m)`，讓與參考片段的平均標準化距離更大。\n\n**跨測站分析** 以路程推估延遲，再用 DTW 與 Pearson 相關檢查可能的時空關係。""")
    c1,c2,c3=st.columns(3)
    c1.metric("總筆數",n);c2.metric("人工異常起點（0 起算）",ab[0]);c3.metric("人工異常長度",ab[1]-ab[0])
    show_series(y,ab,title="先看資料：紅色區塊是我們人工植入的答案")
    st.success("操作順序：先按第 1 頁改視窗大小，再到第 2 頁改起點與長度，再到第 3 頁看自動搜尋。")
with tabs[1]:
    st.subheader("Stage 1：Matrix Profile + STOMP")
    st.write("**MP 在做什麼？** 每個固定長度片段，都找最近的不重疊鄰居。最近鄰居都差很多 ⇒ MP 值高 ⇒ 較可疑。**STOMP** 利用滑動點積的遞推式加速計算。")
    m1=st.slider("固定視窗 w（所有初始候選都這麼長）",4,min(24,ppd*2),ppd)
    k=st.slider("要找幾個候選 Top-k？",1,5,3)
    profile,neighbor=stomp_profile(tuple(y),m1)
    tops=topk_indices(profile,k,m1)
    st.session_state["mp_last"]={"n":n,"m":m1,"top":tops}
    sel=st.selectbox("選擇一個 Top-k 候選查看",list(range(len(tops))),format_func=lambda q:f"Top-{q+1}：起點 {tops[q]}，MP 距離 {profile[tops[q]]:.3f}") if tops else None
    if sel is not None:
        i=tops[sel];j=neighbor[i]
        show_series(y,ab,(i,m1),i,title="藍虛線：MP 起點；綠色：固定長度候選；紅色：人工異常")
        a,b,c=st.columns(3);a.metric("MP 起點",i);b.metric("最近鄰居起點",int(j));c.metric("MP 距離",f"{profile[i]:.3f}")
        st.write(f"MP 的意思：片段 {i}～{i+m1-1} 找到最像自己的非平凡匹配片段 {j}～{j+m1-1}，兩者標準化歐氏距離為 {profile[i]:.3f}。")
    fig=go.Figure(go.Scatter(x=np.arange(len(profile)),y=profile,mode="lines",name="MP 距離"))
    for idx in tops:fig.add_vline(x=idx,line_color="crimson",line_dash="dash",opacity=.7)
    fig.update_layout(title="Matrix Profile（紅線：Top-k 起點）",xaxis_title="片段起始索引",yaxis_title="最近鄰居距離（未除 √m）",height=310)
    st.plotly_chart(fig,width="stretch")
    with st.expander("🧮 STOMP 為什麼快？點我看實際數字"):
        z=[1,2,3,4,5,6]
        st.code("舊： [1,2,3] · [3,4,5] = 1×3 + 2×4 + 3×5 = 26\n新： [2,3,4] · [4,5,6] = 26 - (1×3) + (4×6) = 47",language="text")
        st.write("不必重新計算整個內積。**但整體 STOMP 仍是 O(n²)**，不是 O(n)。這一頁實作的是 STOMP 的點積遞推版本。")
    st.caption("本工具 Top-k 額外採非重疊篩選以方便教學。論文 Stage 1 的核心是 MP 高分候選；各種 Top-k 重疊處理策略可能影響結果。")
with tabs[2]:
    st.subheader("Stage 2（上半段）：LHS 比較樣本與異常分數")
    st.markdown("**在這頁，你手動當 CSOS。** 改 `(i,m)` 讓分數變大。紅色是事先植入答案；綠色是你正在測試的解。分數越大，只代表跟抽樣參考片段越不像，**不保證更貼近真實答案**。")
    mp_default=st.session_state.get("mp_last")
    anchor=mp_default["top"][0] if mp_default and mp_default["n"]==n else ab[0]
    anchor=int(np.clip(anchor,0,n-4))
    radius=st.slider("LHS 周圍參考時間範圍 ± 筆",ppd,min(n//2,ppd*8),min(ppd*4,n//2),step=ppd)
    q=st.slider("LHS 抽樣數 q",3,18,8)
    move=st.slider("起點搜尋半徑 ± 筆",2,ppd*3,ppd)
    low_i=max(0,anchor-move);high_i=min(n-4,anchor+move)
    low_m=3;high_m=min(ppd*2,n-high_i)
    st.caption(f"使用 Stage 1 的候選起點 anchor={anchor}；起點搜尋範圍 {low_i}～{high_i}。")
    i=st.slider("手動選異常起點 i",low_i,high_i,int(np.clip(ab[0],low_i,high_i)))
    max_len=min(ppd*2,n-i)
    m=st.slider("手動選異常長度 m",3,max_len,int(np.clip(duration,3,max_len)))
    refs=lhs_positions(n,anchor,radius,q,3,seed)
    score,used=score_candidate(y,i,m,refs)
    refscore,refcount=score_candidate(y,anchor,min(ppd,n-anchor),refs)
    c1,c2,c3=st.columns(3)
    c1.metric("你選的 score",f"{score:.3f}" if score>-1e8 else "無有效比較")
    c2.metric("初始候選 score",f"{refscore:.3f}" if refscore>-1e8 else "無有效比較")
    c3.metric("本次有效比較樣本",f"{used}/{len(refs)}")
    show_series(y,ab,(i,m),anchor,title="試著對齊紅色異常，再比較分數是否真的最大")
    st.write("**LHS 參考片段起點（不是候選解）：**",refs)
    st.caption("為公平比較，抽樣位置對同一設定固定；每個候選只使用合法且不重疊的參考片段，因此不同長度的有效比較數可能不同。")
    if score>-1e8:
        st.latex(r"score(i,m)=\frac{1}{q}\sum_j \frac{\|z(T_{i,m})-z(S_j)\|_2}{\sqrt m}")
        st.write(f"現在 `i={i}, m={m}`：平均距離 {score:.3f}，最佳化目標值 **-score={-score:.3f}**。")
    with st.expander("📚 為何與真實異常完全重疊，分數也可能下降？"):
        st.write("因為目標函數量的是『跟參考片段的平均形狀差異』，不是與人工標籤的重疊率。Z-normalization、參考樣本與候選長度都會影響結果；這是論文代理指標的限制。")
with tabs[3]:
    st.subheader("Stage 2（下半段）：讓 CSOS 自動找 (i,m)")
    st.markdown("HDBSCAN 把 `(i,m)` 候選解分群；CSOS 使用 **Mutualism（兩者更新）→ Commensalism（單方更新）→ Parasitism（跨群突變）** 嘗試改善異常分數。")
    st.warning("本頁是可執行的 CSOS 教學實作（固定初始群組＋整數邊界），**不是作者原始碼重現**。若已安裝 scikit-learn，才會使用真正的 HDBSCAN；否則明確顯示單群備援。")
    anchor=int((st.session_state.get("mp_last") or {}).get("top",[ab[0]])[0])
    anchor=max(0,min(anchor,n-4)); radius=min(n//2,ppd*4);q=8
    refs=lhs_positions(n,anchor,radius,q,3,seed)
    c1,c2,c3=st.columns(3)
    popsize=c1.slider("Agent 數 p",10,60,28,step=2)
    maxfe=c2.slider("最多評估次數（教學上限）",100,3000,850,step=50)
    spread=c3.slider("搜尋半徑",3,ppd*3,ppd)
    bounds=((max(0,anchor-spread),min(n-4,anchor+spread)),(3,min(2*ppd,n-max(0,anchor-spread))))
    if st.button("🚀 執行 CSOS 搜尋",type="primary",width="stretch"):
        result=run_csos(tuple(y),tuple(refs),bounds,seed,popsize,maxfe)
        signature=(tuple(np.round(y,6)),tuple(refs),bounds,seed,popsize,maxfe)
        st.session_state["csos_result"]=(signature,result)
    saved=st.session_state.get("csos_result")
    current_signature=(tuple(np.round(y,6)),tuple(refs),bounds,seed,popsize,maxfe)
    if saved and saved[0]==current_signature:
        best,score,history,labels,method,snapshots,evals=saved[1]
        a,b,c=st.columns(3);a.metric("找到的起點",best[0]);b.metric("找到的長度",best[1]);c.metric("找到的異常分數",f"{score:.3f}")
        st.info("分群方式："+method)
        show_series(y,ab,best,anchor,title="CSOS 找到的候選（綠）與真實人工異常（紅）")
        fig=go.Figure(go.Scatter(y=[-v for v in history],mode="lines+markers",name="Best score"))
        fig.update_layout(title="最佳分數隨迭代變化",xaxis_title="迭代",yaxis_title="Best anomaly score",height=280)
        st.plotly_chart(fig,width="stretch")
        snapid=st.slider("查看 Agent 分布：從初始化到後期",0,len(snapshots)-1,len(snapshots)-1)
        snapshot=snapshots[snapid]
        fig=go.Figure(go.Scatter(x=snapshot[:,0],y=snapshot[:,1],mode="markers",marker=dict(color=labels,colorscale="Viridis",size=10,showscale=False),text=[f"Agent {j}" for j in range(len(snapshot))]))
        fig.add_trace(go.Scatter(x=[best[0]],y=[best[1]],mode="markers",marker=dict(color="red",symbol="star",size=17),name="Best"))
        fig.update_layout(title="每點是一個 Agent；X=起點，Y=長度",xaxis_title="i",yaxis_title="m",height=340)
        st.plotly_chart(fig,width="stretch")
        st.caption(f"已評估 {evals} 組不同整數候選（使用快取，並非論文的 MaxFE 100,000 次函數呼叫口徑）。")
    else:
        st.info("按上面的「執行 CSOS 搜尋」即可看到搜尋出的起點、長度、分數曲線和 Agent 分布。")
with tabs[4]:
    st.subheader("跨測站：DTW + Pearson correlation")
    st.write("做一個『上游先異常，下游稍晚出現相似形狀』的教學情境。先設延遲搜尋範圍，再用 DTW 找到最相似位置，最後用 Pearson 看模式是否同向變化。")
    lag=st.slider("人工設計的下游延遲（筆）",2,20,8)
    jitter=st.slider("下游測站雜訊",0.,2.,0.45,.05)
    rng=np.random.default_rng(seed+87)
    down=np.roll(y,lag)+rng.normal(0,jitter,n)
    down[:lag]=base[:lag]+rng.normal(0,jitter,lag)
    m0=max(3,ab[1]-ab[0]);src=y[ab[0]:ab[0]+m0]
    estimated=lag;lo=max(0,ab[0]+estimated-4);hi=min(n-m0,ab[0]+estimated+5)
    results=[]
    for t in range(lo,hi+1):
        cand=down[t:t+m0]
        cor=float(np.corrcoef(src,cand)[0,1]) if np.std(src)>1e-10 and np.std(cand)>1e-10 else 0.0
        results.append((t,dtw(src,cand),cor))
    results=sorted(results,key=lambda a:a[1]);best=results[0]
    show_series(down,(ab[0]+lag,ab[1]+lag),(best[0],m0),title="下游測站：紅色是真實延遲異常，綠色是 DTW 找到的候選")
    a,b,c=st.columns(3)
    a.metric("DTW 最佳起點",best[0]);b.metric("DTW 距離（越小越相似）",f"{best[1]:.3f}");c.metric("Pearson 相關",f"{best[2]:.3f}")
    if best[2]>.5:st.success("相關係數 > 0.5：符合論文採用的相關門檻（不代表已證明因果傳播）。")
    else:st.warning("相關係數 ≤ 0.5：未通過本示範的相關門檻。")
    st.dataframe(pd.DataFrame(results,columns=["下游候選起點","DTW 距離","Pearson 相關"]).sort_values("DTW 距離").round(4),hide_index=True,width="stretch")
    st.caption("教學版下游資料由上游資料延遲複製加雜訊而成，不代表真實車輛沿高速公路的物理傳播。論文另以里程、速度建立時窗並分析真實測站。")
with tabs[5]:
    st.subheader("論文 Table 2：為什麼 Stage 2 看起來更快？")
    paper=pd.DataFrame({"資料筆數":["0.5M","0.8M","1M"],"Stage 1 STOMP（秒）":[83.75,209.42,349.84],"Stage 2 CSOS Top-1（秒）":[7.32,7.44,7.45],"Stage 2 CSOS Top-5（秒）":[37.22,37.23,37.27],"Stage 2 CSOS Top-10（秒）":[74.04,73.98,74.01]})
    st.dataframe(paper,hide_index=True,width="stretch")
    st.markdown("""**Stage 1** 在完整資料上建立 MP，資料越長，時間明顯增加。**Stage 2** 固定只最佳化 Top-k 個候選、相同搜尋參數，時間主要隨 k 成長，與原始資料長度 n 關係較小。**LHS** 減少每次候選評估的比較數量；**但論文沒有單獨移除 LHS 的消融實驗，不能說表中速度全是 LHS 的功勞。**""")
    fig=go.Figure()
    for key in ["Stage 1 STOMP（秒）","Stage 2 CSOS Top-10（秒）"]:
        fig.add_trace(go.Bar(name=key,x=paper["資料筆數"],y=paper[key]))
    fig.update_layout(barmode="group",height=340,yaxis_title="秒")
    st.plotly_chart(fig,width="stretch")
    st.subheader("論文 Table 1：Top-10 相對異常分數準確率（不是分類 Accuracy）")
    accuracy=pd.DataFrame({"年份":[2017,2018,2019],"LRRDS":[67.70,71.20,81.50],"MAD":[93.20,92.80,92.60],"SSP":[90.76,90.09,89.97],"MP_OPT":[96.20,94.90,94.70]})
    st.dataframe(accuracy,hide_index=True,width="stretch")
    st.markdown("**限制：** 這是異常分數相較抽樣片段的排名比例，不是人工標籤下的異常辨識率。")
    st.subheader("Meeting 四句話總結")
    st.markdown("""1. **MP** 對全時間序列算固定長度最近鄰居距離，**STOMP** 重用點積加速。\n2. 從 MP 中選出 **Top-k 候選**，不代表整份資料只有 k 個異常。\n3. **LHS** 選參考片段、**HDBSCAN** 分群候選解、**CSOS** 最佳化 `(i,m)`。\n4. **DTW + Pearson** 分析不同測站可能的時空關聯；最高分不保證等於真實異常邊界。""")
    csv=pd.DataFrame({"index":np.arange(n),"vehicle_count":y,"injected_anomaly":((np.arange(n)>=ab[0])&(np.arange(n)<ab[1])).astype(int)}).to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 下載本次實驗原始資料 CSV",csv,"mpopt_experiment.csv","text/csv")
