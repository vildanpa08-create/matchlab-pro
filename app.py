import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from upset_finder import load_matches, ALL_LEAGUES, BOOKS
from score_analysis import render_streamlit
st.set_page_config(page_title='MATCHLAB PRO V3',layout='wide')
st.title('MATCHLAB PRO V3')
st.caption('Geçmiş oran eşleştirme ve sonuç analizi')
@st.cache_data(ttl=3600)
def load(leagues,seasons,book):
    return load_matches(leagues, seasons, book), []
with st.sidebar:
    source=st.radio('Veri kaynağı',['Football-data','CSV yükle'])
    leagues=st.multiselect('Ligler',list(ALL_LEAGUES),format_func=lambda x: f'{x} - {ALL_LEAGUES[x]}',default=['E0','D1','SP1','I1','F1','T1'])
    seasons=st.multiselect('Sezonlar',[f'{y%100:02d}{(y+1)%100:02d}' for y in range(2016,2027)],default=['2324','2425','2526'])
    book=st.selectbox('Oran kaynağı',[b for b,_ in BOOKS],format_func=lambda b:dict(BOOKS)[b])
    upload=st.file_uploader('CSV (date, home, away, oh, od, oa, hg, ag; isteğe bağlı ht_h, ht_a, book)',type='csv') if source=='CSV yükle' else None
with st.form('search'):
    cols=st.columns(3)
    h=cols[0].number_input('MS 1',min_value=1.01,value=6.0)
    d=cols[1].number_input('MS X',min_value=1.01,value=5.75)
    a=cols[2].number_input('MS 2',min_value=1.01,value=1.38)
    tol=st.slider('Tolerans (%)',1,25,5)/100
    go=st.form_submit_button('Analiz et',use_container_width=True)
if go:
    try:
        if source=='Football-data':
            if not leagues or not seasons: st.warning('Lig ve sezon seçin.'); st.stop()
            with st.spinner('Veriler yükleniyor…'): df,missing=load(tuple(leagues),tuple(seasons),book)
            st.caption('Arşiv yalnızca seçilen oran kaynağında bulunan kayıtları içerir; ek liglerde sezon filtresi başlangıç yılına göre uygulanır.')
        else:
            if upload is None: st.stop()
            df=pd.read_csv(upload)
        if df.empty: st.warning('Veri bulunamadı.'); st.stop()
        for c in ['oh','od','oa','hg','ag','ht_h','ht_a']:
            df[c]=pd.to_numeric(df[c],errors='coerce') if c in df else np.nan
        df=df.dropna(subset=['oh','od','oa','hg','ag','home','away'])
        df=df[(df[['oh','od','oa']]>1).all(axis=1)&(df[['hg','ag']]>=0).all(axis=1)]
        df['date']=pd.to_datetime(df['date'],errors='coerce',dayfirst=True)
        df=df.drop_duplicates(subset=['date','home','away','oh','od','oa'])
        df['ftr']=np.where(df.hg>df.ag,'H',np.where(df.hg<df.ag,'A','D'))
        ht_valid=df.ht_h.notna() & df.ht_a.notna()
        if 'htr' not in df: df['htr']=None
        df.loc[ht_valid,'htr']=np.where(df.loc[ht_valid,'ht_h']>df.loc[ht_valid,'ht_a'],'H',np.where(df.loc[ht_valid,'ht_h']<df.loc[ht_valid,'ht_a'],'A','D'))
        s=df.copy(); dist=[]
        for c,v in [('oh',h),('od',d),('oa',a)]:
            s=s[s[c].between(v*(1-tol),v*(1+tol))]
        s=s.copy(); s['distance']=sum(abs(s[c]/v-1) for c,v in [('oh',h),('od',d),('oa',a)])/3
        s=s.sort_values('distance'); st.session_state['result']=s; st.session_state['total']=len(df); st.session_state['analysis_id']=st.session_state.get('analysis_id',0)+1
    except Exception as e: st.error(str(e)); st.stop()
if 'result' in st.session_state:
    s=st.session_state.result; n=len(s)
    st.write(f'Arşiv: {st.session_state.total:,} kayıt | Eşleşme: {n:,}')
    if not n: st.warning('Eşleşme yok. Toleransı genişletin.'); st.stop()
    if n<100: st.warning('Küçük örneklem: sonuçlar yüksek belirsizlik içerir.')
    cols=st.columns(3)
    for col,k,label in zip(cols,['H','D','A'],['MS 1','MS X','MS 2']): col.metric(label,f'{(s.ftr==k).mean():.1%}')
    st.subheader('Gol dağılımı')
    goals=s.hg+s.ag
    stats=pd.DataFrame({'Market':[f'{i+.5} Üst' for i in range(6)]+['KG Var'],'Gerçekleşme':[float((goals>i+.5).mean()) for i in range(6)]+[float(((s.hg>0)&(s.ag>0)).mean())]})
    st.bar_chart(stats.set_index('Market'))
    st.subheader('Maç sonu skorları'); st.dataframe(s.groupby(['hg','ag']).size().sort_values(ascending=False).rename('Adet').reset_index())
    ht=s.dropna(subset=['ht_h','ht_a']).copy()
    st.subheader(f'İlk yarı / maç sonu ({len(ht)} kayıt)')
    if len(ht):
        ht['İY']=np.where(ht.ht_h>ht.ht_a,'1',np.where(ht.ht_h<ht.ht_a,'2','0')); ht['MS']=ht.ftr.map({'H':'1','D':'0','A':'2'})
        st.dataframe(ht.groupby(['İY','MS']).size().sort_values(ascending=False).rename('Adet').reset_index()); st.dataframe(ht.groupby(['ht_h','ht_a']).size().sort_values(ascending=False).rename('Adet').reset_index())
    else: st.info('İlk yarı verisi yok.')
    render_streamlit(s, key='scores_'+str(st.session_state.get('analysis_id',0)))
    st.subheader('En benzer maçlar'); st.dataframe(s.head(100),use_container_width=True)
    buf=BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer: s.to_excel(writer,index=False,sheet_name='Eşleşmeler'); stats.to_excel(writer,index=False,sheet_name='Gol dağılımı')
    st.download_button('Excel indir',buf.getvalue(),'matchlab_sonuclar.xlsx')
    st.caption('Tarihsel oran eşleşmeleri gelecekteki sonuçları garanti etmez. Gol dağılımları 1X2 eşleşmelerinden hesaplanır; gol marketi oran eşleşmesi değildir.')
