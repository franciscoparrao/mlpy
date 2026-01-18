# Guía de Humanización del Paper MLPY

**Objetivo**: Reescribir el paper académico para que suene natural, escrito por un humano, evitando patrones detectables de LLM.

**Fecha creación**: 2025-10-27
**Versión**: 1.0

---

## 🎯 Principio Fundamental

> **"Un humano experto escribe con confianza irregular, no con perfección mecánica"**

Los papers escritos por humanos tienen:
- Variación natural en estructura de oraciones
- Pequeñas "imperfecciones" que dan personalidad
- Transiciones menos pulidas pero más naturales
- Énfasis variable (no todo es igualmente importante)
- Preferencias estilísticas personales inconsistentes

---

## 🚫 Patrones LLM a EVITAR

### 1. **Listas Excesivamente Balanceadas**

❌ **Estilo LLM** (demasiado perfecto):
```
MLPY provides three key benefits:
- Enhanced modularity for component reuse
- Improved flexibility for complex workflows
- Superior extensibility for custom implementations
```

✅ **Estilo Humano** (natural, desbalanceado):
```
MLPY's modular design makes it easy to compose complex workflows from
reusable components, and the framework can be extended with custom
learners, measures, or pipeline operations without much effort.
```

**Por qué**: Los humanos no balancean listas perfectamente. Algunas ideas se expanden más que otras.

---

### 2. **Transiciones Mecánicas**

❌ **Estilo LLM**:
```
In contrast to traditional approaches, MLPY adopts...
Building upon this foundation, we introduce...
Having established the architecture, we now turn to...
```

✅ **Estilo Humano**:
```
Unlike sklearn, MLPY uses...
The framework also includes...
We evaluate this approach in Section 5.
```

**Por qué**: Los humanos usan transiciones más directas, menos "pulidas".

---

### 3. **Paralelismo Excesivo**

❌ **Estilo LLM** (estructura repetitiva):
```
scikit-learn provides basic functionality but lacks modularity.
PyCaret provides AutoML features but lacks flexibility.
H2O provides distributed computing but lacks extensibility.
```

✅ **Estilo Humano** (variado):
```
While scikit-learn is the standard for ML in Python, its monolithic
design makes composition difficult. PyCaret simplifies AutoML but
sacrifices flexibility, and H2O excels at distributed computing
but has limited extensibility.
```

**Por qué**: Los humanos evitan repetir exactamente la misma estructura.

---

### 4. **Adjetivos Amplificadores Repetidos**

❌ **Estilo LLM**:
```
- significantly improves performance
- substantially reduces code
- dramatically enhances usability
- considerably simplifies workflows
```

✅ **Estilo Humano** (más variado y moderado):
```
- improves performance
- cuts code by 40%
- makes workflows simpler
- reduces boilerplate
```

**Por qué**: Los LLMs usan demasiados intensificadores. Los humanos son más directos.

---

### 5. **Frases "De Libro de Texto"**

❌ **Estilo LLM**:
```
"It is worth noting that..."
"It is important to emphasize that..."
"It should be noted that..."
"One must consider that..."
```

✅ **Estilo Humano**:
```
"Note that..."
"Importantly,..."
Simplemente omitir la frase
Integrar la idea directamente
```

**Por qué**: Los humanos son más concisos. Estas frases agregan palabras sin valor.

---

### 6. **Numeración Perfecta de Todo**

❌ **Estilo LLM** (enumera todo):
```
We make three contributions: (1) unified abstractions,
(2) flexible pipelines, and (3) spatial ML support.

The system has four components: (1) tasks, (2) learners,
(3) measures, and (4) predictions.
```

✅ **Estilo Humano** (mezcla formatos):
```
Our main contributions are unified abstractions for tasks and
learners, a graph-based pipeline system, and native support for
spatial machine learning.

The framework includes tasks, learners, measures, and predictions.
```

**Por qué**: No todo necesita numeración explícita.

---

### 7. **Conclusiones Resumiendo Todo**

❌ **Estilo LLM**:
```
In summary, we have presented MLPY, a modular framework that
provides unified abstractions, flexible pipelines, spatial ML
support, and data validation, achieving comparable performance
while offering superior modularity.
```

✅ **Estilo Humano**:
```
MLPY brings mlr3's modular design to Python, with native support
for spatial ML—a feature missing in other frameworks.
```

**Por qué**: Los humanos no repiten toda la información. Eligen 1-2 puntos clave.

---

### 8. **Pasiva Académica Excesiva**

❌ **Estilo LLM**:
```
The model was trained using cross-validation.
Results were obtained through extensive experimentation.
Performance was evaluated across multiple metrics.
```

✅ **Estilo Humano** (mezcla activa/pasiva):
```
We trained the model using cross-validation.
The results show that...
We evaluate performance with accuracy and F1 score.
```

**Por qué**: Papers modernos mezclan voz activa y pasiva. "We" es aceptable en papers.

---

## ✅ Técnicas de Humanización

### Técnica 1: **Variación en Longitud de Oraciones**

Alterna oraciones cortas, medianas y largas:

```
✅ Ejemplo:
"MLPY is modular. Unlike scikit-learn's monolithic design, it
separates tasks, learners, and measures into independent components
that can be composed freely. This makes complex workflows simpler."

Corta → Larga → Media
```

---

### Técnica 2: **Desequilibrio Intencional**

No todo merece la misma cantidad de palabras:

```
✅ Ejemplo:
"The framework includes three task types: classification and
regression (supervised learning), and clustering. Time series
and spatial tasks extend these base classes."

Primera parte: detallada
Segunda parte: breve mención
```

---

### Técnica 3: **Imperfecciones Naturales**

Pequeñas repeticiones o redundancias que un humano dejaría:

```
✅ Ejemplo:
"Regular CV randomly assigns observations to folds, which means
nearby observations can end up in both training and test sets.
This random assignment allows the model to exploit spatial structure."

Nota: "randomly" y "random assignment" - un humano podría dejar esto
```

---

### Técnica 4: **Opiniones Sutiles**

Los humanos expresan preferencias:

```
✅ Ejemplos:
"This is particularly problematic for..."
"Unfortunately, sklearn lacks..."
"A more elegant solution is..."
"Interestingly, the error difference..."
```

---

### Técnica 5: **Conectores Informales en Contexto Formal**

Usar ocasionalmente conectores menos formales:

```
✅ Ejemplos:
"And yet, regular CV still..."
"But this approach fails when..."
"So we need..."
```

---

## 📝 Proceso de Reescritura (Paso a Paso)

### Paso 1: Leer Sección Completa (5 min)
- Leer versión ES y EN actuales
- Identificar mensaje clave de cada párrafo
- Marcar frases que suenan "demasiado LLM"

### Paso 2: Reescribir Párrafo por Párrafo (40-60 min por sección)

Para cada párrafo:

1. **Extraer la idea central** (1 frase)
2. **Escribir primera oración** de forma directa
3. **Desarrollar idea** con variación natural
4. **Terminar párrafo** sin resumir todo

**Preguntas guía**:
- ¿Cómo explicaría esto verbalmente a un colega?
- ¿Qué parte es realmente importante vs. relleno?
- ¿Usaría estas palabras exactas al hablar?

### Paso 3: Revisar Sección Completa (10 min)

Checklist:
- [ ] ¿Varían las longitudes de oraciones?
- [ ] ¿Hay listas excesivamente balanceadas?
- [ ] ¿Suenan las transiciones demasiado pulidas?
- [ ] ¿Usé muchos "significantly/substantially"?
- [ ] ¿Leería esto en voz alta sin sentirme raro?

### Paso 4: Test de Lectura en Voz Alta (5 min)

Lee 2-3 párrafos en voz alta. Si suena como un robot, reescribe.

---

## 🎨 Perfiles de Estilo Aceptables

### Estilo A: "Directo y Técnico"
- Oraciones cortas
- Énfasis en datos y resultados
- Menos contexto narrativo
- Más tablas y números

### Estilo B: "Narrativo pero Preciso"
- Oraciones medianas a largas
- Contextualiza decisiones
- Explica "por qué" además de "qué"
- Balance entre prosa y datos

**Para MLPY**: Usaremos **Estilo B** (narrativo) en intro/motivación, **Estilo A** (directo) en resultados.

---

## 📊 Antes/Después: Ejemplos Completos

### Ejemplo 1: Abstract

❌ **Versión LLM** (160 palabras):
```
Machine learning in Python is dominated by monolithic frameworks that,
while powerful, lack the modularity and composability needed for modern
ML workflows. We present MLPY, a modular and extensible framework inspired
by mlr3's design principles, bringing object-oriented composability to
Python's ML ecosystem. MLPY provides unified abstractions for tasks,
learners, measures, and resampling strategies, enabling complex ML
workflows with minimal code. Unlike existing frameworks, MLPY includes
native support for spatial machine learning with spatially-aware
cross-validation and automatic data validation. Benchmarks show MLPY
achieves comparable performance to scikit-learn while providing a more
modular and flexible API. The framework is open source under Apache 2.0
license and available at https://github.com/franciscoparrao/mlpy.
```

✅ **Versión Humanizada** (155 palabras):
```
Python's machine learning ecosystem is dominated by scikit-learn, a
powerful but monolithic framework where preprocessing, modeling, and
evaluation are tightly coupled. This makes it difficult to compose
complex pipelines from reusable components or extend the framework
cleanly. We present MLPY, a modular framework inspired by R's mlr3
that brings object-oriented composability to Python. MLPY separates
concerns through Task, Learner, Measure, and Prediction abstractions,
enabling complex workflows with less code than sklearn. The framework
includes native support for spatial machine learning—currently missing
in Python—with spatially-aware cross-validation that prevents inflated
performance estimates on geographic data. We show that regular CV
underestimates prediction error by 19% on spatial data, while MLPY's
spatial resampling provides realistic estimates. MLPY achieves
comparable performance to sklearn while offering better modularity.
The framework is open source (Apache 2.0) at github.com/franciscoparrao/mlpy.
```

**Cambios clave**:
- Menos listas perfectas
- Más detalles concretos (19%)
- Conectores más naturales ("This makes it...", "We show that...")
- Variación en longitud de oraciones

---

### Ejemplo 2: Sección Spatial ML - Párrafo Introductorio

❌ **Versión LLM**:
```
When machine learning methods are applied to spatial or geographic data,
standard cross-validation methods can produce overly optimistic performance
estimates. This occurs because spatially proximal observations tend to be
more similar than distant ones—a phenomenon known as spatial autocorrelation.
Regular cross-validation randomly assigns nearby observations to both
training and test sets, allowing the model to exploit this spatial structure
and achieve artificially high performance that does not generalize to
spatially distant regions.
```

✅ **Versión Humanizada**:
```
Spatial data breaks a key assumption of standard cross-validation: that
observations are independent. In reality, nearby locations tend to have
similar values—a phenomenon called spatial autocorrelation. When regular
CV randomly assigns observations to folds, it scatters nearby points across
training and test sets. The model then exploits spatial structure to achieve
high accuracy on the test set, but this performance doesn't generalize to
new geographic regions. The result is overly optimistic error estimates.
```

**Cambios clave**:
- Primera oración más directa: "breaks a key assumption"
- Menos pasiva: "When regular CV randomly assigns" → "it scatters"
- Última oración cortita para énfasis
- Más coloquial: "In reality" en lugar de "This occurs because"

---

## 🔍 Checklist Final (Antes de Considerar Terminado)

### Nivel Párrafo
- [ ] Las oraciones varían en longitud (10-30 palabras)
- [ ] No hay más de 2 oraciones con estructura similar consecutivas
- [ ] Los conectores son variados (no siempre "However," "Therefore,")
- [ ] Hay al menos 1 oración corta (< 12 palabras) por cada 4-5 oraciones

### Nivel Sección
- [ ] Las listas tienen 2-4 items (no siempre 3)
- [ ] No todas las comparaciones usan la misma estructura
- [ ] Los adjetivos amplificadores aparecen < 3 veces por página
- [ ] Hay mezcla de voz activa ("We show") y pasiva ("Results indicate")

### Nivel Paper
- [ ] El abstract no enumera todas las contribuciones
- [ ] La introducción no dice "The rest of this paper is organized as follows"
- [ ] La conclusión no resume todo el paper
- [ ] Hay variación entre secciones (algunas más técnicas, otras más narrativas)

---

## 🎯 Estrategia de Reescritura por Sección

### Prioridad 1: Secciones Críticas (80% del impacto)
1. **Abstract** (20 min) - Primera impresión
2. **Introduction** (30 min) - Contexto y motivación
3. **Spatial ML Case Study** (60 min) - Contribución principal
4. **Conclusion** (15 min) - Mensaje final

### Prioridad 2: Secciones Técnicas (15% del impacto)
5. **AutoML Tuning** (40 min)
6. **Architecture** (30 min)
7. **Implementation** (20 min)

### Prioridad 3: Secciones Estándar (5% del impacto)
8. **Related Work** (15 min)
9. **Examples** (15 min)

**Total estimado**: 4-5 horas de reescritura concentrada

---

## 💾 Workflow de Guardado

Para no perder trabajo:

1. **Crear backup** antes de empezar cada sección:
   ```bash
   cp section_spatial_ml_case_study.tex section_spatial_ml_case_study.tex.backup
   ```

2. **Commit git después** de cada sección terminada:
   ```bash
   git add section_spatial_ml_case_study.tex
   git commit -m "Humanize: Spatial ML section rewrite"
   ```

3. **Compilar PDF** después de cada sección para verificar LaTeX:
   ```bash
   pdflatex mlpy_jmlr_paper_FINAL.tex
   ```

---

## 📚 Recursos de Referencia

### Papers Académicos Bien Escritos (para inspirarse)
1. Original scikit-learn paper (Pedregosa et al., 2011)
2. mlr3 paper (Lang et al., 2019)
3. Papers de JMLR vol. 20-24 (estilo moderno)

### Qué buscar en esos papers:
- Cómo varían la estructura
- Cuándo usan listas vs. prosa
- Cómo escriben el abstract
- Qué tan perfectas son las transiciones (spoiler: no mucho)

---

## 🚀 Inicio Rápido (TL;DR)

Si pierdes contexto y necesitas retomar:

1. **Abre esta guía** y lee "Patrones LLM a EVITAR"
2. **Lee la sección EN actual** y marca frases muy LLM
3. **Reescribe párrafo por párrafo** preguntándote: "¿Cómo diría esto verbalmente?"
4. **Aplica técnicas**: Varía longitud, rompe paralelismo, sé más directo
5. **Lee en voz alta** - si suena robótico, reescribe
6. **Compila LaTeX** para verificar que no rompiste nada
7. **Commit git** con mensaje claro

---

## ⚠️ Advertencias

### NO hagas esto:
❌ Reemplazar palabras con sinónimos al azar (suena peor)
❌ Agregar errores gramaticales a propósito (no es lo mismo)
❌ Cambiar terminología técnica (confunde al lector)
❌ Eliminar precisión por "sonar humano" (es un paper académico)

### SÍ haz esto:
✅ Variar estructura naturalmente
✅ Ser más directo y menos pulido
✅ Desequilibrar intencionalmente
✅ Usar conectores informales ocasionalmente
✅ Mantener precisión técnica

---

## 📈 Métricas de Éxito

Un párrafo está bien humanizado cuando:

1. **Test de lectura en voz alta**: Suena como algo que dirías
2. **Test de variación**: No hay 3 oraciones seguidas con estructura similar
3. **Test de simpleza**: Prefieres la versión nueva a la antigua
4. **Test de precisión**: No perdiste información técnica

---

**Última actualización**: 2025-10-27
**Autor**: Claude (para Francisco Parra)
**Estado**: Guía completa v1.0 - Lista para usar

