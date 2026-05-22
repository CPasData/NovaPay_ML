
-- ESQUEMA DE BASE DE DATOS — DETECCIÓN DE FRAUDE
-- Sistema: NovaPay ML
-- Motor:  SQL Server (T-SQL)



-- 1. CLIENTES

CREATE TABLE Clientes (
    id_cliente          VARCHAR(12)     NOT NULL,
    tipo_cliente        VARCHAR(20)     NOT NULL,
    edad_cliente        TINYINT         NOT NULL,
    customer_country    CHAR(2)         NOT NULL,
    customer_region     VARCHAR(50)     NOT NULL,
    tenure              INT             NOT NULL,
    importe_medio_mensual       DECIMAL(12,2)   NOT NULL,
    desviacion_estandar_mensual DECIMAL(12,2)   NOT NULL,
    media_transacciones_al_dia  DECIMAL(6,2)    NOT NULL,
    numero_fraudes_ultimo_ano   INT             NOT NULL DEFAULT 0,

    CONSTRAINT PK_Clientes PRIMARY KEY (id_cliente),
    CONSTRAINT CK_Clientes_tipo CHECK (tipo_cliente IN ('persona', 'empresa', 'autónomo', 'premium')),
    CONSTRAINT CK_Clientes_edad CHECK (edad_cliente BETWEEN 18 AND 85),
    CONSTRAINT CK_Clientes_tenure CHECK (tenure >= 0),
    CONSTRAINT CK_Clientes_importe_medio CHECK (importe_medio_mensual >= 0)
);

CREATE INDEX IX_Clientes_pais ON Clientes (customer_country);
CREATE INDEX IX_Clientes_tipo ON Clientes (tipo_cliente);


-- 2. CUENTAS (origen)

CREATE TABLE Cuentas (
    id_cuenta           VARCHAR(12)     NOT NULL,
    id_cliente          VARCHAR(12)     NOT NULL,
    cuenta_origen       VARCHAR(34)     NOT NULL,
    estado_cuenta       VARCHAR(20)     NOT NULL,
    saldo_actual        DECIMAL(14,2)   NOT NULL,
    saldo_medio_30_dias DECIMAL(14,2)   NOT NULL,
    volumen_entrante_30_dias    DECIMAL(14,2)   NOT NULL,
    volumen_saliente_30_dias    DECIMAL(14,2)   NOT NULL,
    numero_transferencias_recibidas_7_dias INT NOT NULL DEFAULT 0,
    numero_transferencias_enviadas_7_dias   INT NOT NULL DEFAULT 0,

    CONSTRAINT PK_Cuentas PRIMARY KEY (id_cuenta),
    CONSTRAINT FK_Cuentas_Clientes FOREIGN KEY (id_cliente) REFERENCES Clientes(id_cliente),
    CONSTRAINT CK_Cuentas_estado CHECK (estado_cuenta IN ('activa', 'bloqueada', 'suspendida', 'cerrada')),
    CONSTRAINT CK_Cuentas_saldo CHECK (saldo_actual >= 0),
    CONSTRAINT UQ_Cuentas_origen UNIQUE (cuenta_origen)
);

CREATE INDEX IX_Cuentas_cliente ON Cuentas (id_cliente);
CREATE INDEX IX_Cuentas_estado ON Cuentas (estado_cuenta);


-- 3. TARJETAS

CREATE TABLE Tarjetas (
    id_tarjeta                      VARCHAR(12)     NOT NULL,
    id_cuenta                       VARCHAR(12)     NOT NULL,
    id_cliente                      VARCHAR(12)     NOT NULL,
    estado_tarjeta                  VARCHAR(20)     NOT NULL,
    fecha_creacion_tarjeta          DATE            NOT NULL,
    antiguedad_tarjeta_dias         INT             NOT NULL,
    limite_importe_transacciones    DECIMAL(10,2)   NOT NULL,
    veces_superar_limite_7_dias     INT             NOT NULL DEFAULT 0,

    CONSTRAINT PK_Tarjetas PRIMARY KEY (id_tarjeta),
    CONSTRAINT FK_Tarjetas_Cuentas FOREIGN KEY (id_cuenta) REFERENCES Cuentas(id_cuenta),
    CONSTRAINT FK_Tarjetas_Clientes FOREIGN KEY (id_cliente) REFERENCES Clientes(id_cliente),
    CONSTRAINT CK_Tarjetas_estado CHECK (estado_tarjeta IN ('activa', 'bloqueada', 'caducada', 'robada', 'extraviada')),
    CONSTRAINT CK_Tarjetas_limite CHECK (limite_importe_transacciones > 0),
    CONSTRAINT CK_Tarjetas_antiguedad CHECK (antiguedad_tarjeta_dias >= 0)
);

CREATE INDEX IX_Tarjetas_cuenta ON Tarjetas (id_cuenta);
CREATE INDEX IX_Tarjetas_cliente ON Tarjetas (id_cliente);
CREATE INDEX IX_Tarjetas_estado ON Tarjetas (estado_tarjeta);


-- 4. CUENTAS DESTINO (catálogo de beneficiarios)

CREATE TABLE Cuentas_Destino (
    cuenta_destino      VARCHAR(34)     NOT NULL,
    destino_alto_riesgo BIT             NOT NULL DEFAULT 0,
    fecha_alta          DATETIME2       NOT NULL DEFAULT SYSUTCDATETIME(),

    CONSTRAINT PK_Cuentas_Destino PRIMARY KEY (cuenta_destino)
);

CREATE INDEX IX_Destino_riesgo ON Cuentas_Destino (destino_alto_riesgo)
    WHERE destino_alto_riesgo = 1;


-- 5. TRANSACCIONES (tabla principal del modelo ML)

CREATE TABLE Transacciones (
    id_transaccion                      VARCHAR(12)     NOT NULL,
    id_cliente                          VARCHAR(12)     NOT NULL,
    id_cuenta                           VARCHAR(12)     NOT NULL,
    id_tarjeta                          VARCHAR(12)     NULL,
    cuenta_destino                      VARCHAR(34)     NULL,
    -- Timestamps
    fecha_hora                          DATETIME2       NOT NULL,
    fecha_insercion                     DATETIME2       NOT NULL DEFAULT SYSUTCDATETIME(),
    -- Tipo
    tipo_transaccion                    VARCHAR(20)     NOT NULL,
    is_night                            BIT             NOT NULL,
    is_weekend                          BIT             NOT NULL,
    -- Métricas temporales
    tiempo_desde_ultima_transaccion     INT             NOT NULL,
    numero_transacciones_ultima_hora    INT             NOT NULL,
    -- Importe
    importe_transaccion                 DECIMAL(12,2)   NOT NULL,
    -- Autenticación
    metodo_autenticacion                VARCHAR(20)     NOT NULL,
    numero_pin_disponibles              TINYINT         NOT NULL,
    -- Dispositivo
    identificador_dispositivo_fingerprint VARCHAR(32)   NULL,
    dispositivo_reconocido              BIT             NOT NULL DEFAULT 1,
    -- Geolocalización de la operación
    operacion_pais                      CHAR(2)         NOT NULL,
    operacion_region                    VARCHAR(50)     NOT NULL,
    direccion_ip_origen                 VARCHAR(15)     NOT NULL,
    geolocalizacion                     VARCHAR(30)     NOT NULL,
    -- Destino
    destino_alto_riesgo                 BIT             NOT NULL DEFAULT 0,
    -- Etiquetas
    IS_FRAUD                            BIT             NOT NULL DEFAULT 0,
    IMPACTO_FRAUDE                      TINYINT         NOT NULL DEFAULT 0,

    CONSTRAINT PK_Transacciones PRIMARY KEY (id_transaccion),
    CONSTRAINT FK_Transacciones_Clientes FOREIGN KEY (id_cliente) REFERENCES Clientes(id_cliente),
    CONSTRAINT FK_Transacciones_Cuentas  FOREIGN KEY (id_cuenta)  REFERENCES Cuentas(id_cuenta),
    CONSTRAINT FK_Transacciones_Tarjetas FOREIGN KEY (id_tarjeta) REFERENCES Tarjetas(id_tarjeta),
    CONSTRAINT FK_Transacciones_Destino  FOREIGN KEY (cuenta_destino) REFERENCES Cuentas_Destino(cuenta_destino),
    CONSTRAINT CK_Transacciones_tipo CHECK (tipo_transaccion IN ('tarjeta', 'transferencia')),
    CONSTRAINT CK_Transacciones_auth CHECK (metodo_autenticacion IN ('PIN', 'firma', '3DS', 'huella', 'contactless')),
    CONSTRAINT CK_Transacciones_pin CHECK (numero_pin_disponibles BETWEEN 0 AND 3),
    CONSTRAINT CK_Transacciones_importe CHECK (importe_transaccion > 0),
    CONSTRAINT CK_Transacciones_tiempo_ultima CHECK (tiempo_desde_ultima_transaccion >= 0),
    CONSTRAINT CK_Transacciones_ultima_hora CHECK (numero_transacciones_ultima_hora >= 0),
    CONSTRAINT CK_Transacciones_impacto CHECK (IMPACTO_FRAUDE BETWEEN 0 AND 3)
);

-- Índices principales para el modelo ML y reporting
CREATE INDEX IX_Transacciones_cliente      ON Transacciones (id_cliente) INCLUDE (IS_FRAUD);
CREATE INDEX IX_Transacciones_fecha        ON Transacciones (fecha_hora) INCLUDE (IS_FRAUD);
CREATE INDEX IX_Transacciones_tipo         ON Transacciones (tipo_transaccion) INCLUDE (IS_FRAUD);
CREATE INDEX IX_Transacciones_pais         ON Transacciones (operacion_pais) INCLUDE (IS_FRAUD);
CREATE INDEX IX_Transacciones_noche        ON Transacciones (is_night) INCLUDE (IS_FRAUD);
CREATE INDEX IX_Transacciones_fraude       ON Transacciones (IS_FRAUD) WHERE IS_FRAUD = 1;
CREATE INDEX IX_Transacciones_destino_riesgo ON Transacciones (destino_alto_riesgo) INCLUDE (IS_FRAUD);


-- VISTA PARA ML — Planas sin joins

CREATE OR ALTER VIEW vw_Transacciones_ML AS
SELECT
    t.id_transaccion,
    -- Cliente
    c.tipo_cliente,
    c.edad_cliente,
    c.customer_country,
    c.customer_region,
    c.tenure,
    c.importe_medio_mensual,
    c.desviacion_estandar_mensual,
    c.media_transacciones_al_dia,
    c.numero_fraudes_ultimo_ano,
    -- Cuenta origen
    cu.estado_cuenta,
    cu.saldo_actual,
    cu.saldo_medio_30_dias,
    cu.volumen_entrante_30_dias,
    cu.volumen_saliente_30_dias,
    cu.numero_transferencias_recibidas_7_dias,
    cu.numero_transferencias_enviadas_7_dias,
    -- Tarjeta
    tk.estado_tarjeta,
    tk.antiguedad_tarjeta_dias,
    tk.limite_importe_transacciones,
    tk.veces_superar_limite_7_dias,
    -- Transacción
    t.tipo_transaccion,
    t.fecha_hora,
    t.is_night,
    t.is_weekend,
    t.tiempo_desde_ultima_transaccion,
    t.numero_transacciones_ultima_hora,
    t.importe_transaccion,
    t.metodo_autenticacion,
    t.numero_pin_disponibles,
    t.dispositivo_reconocido,
    t.operacion_pais,
    t.operacion_region,
    t.destino_alto_riesgo,
    -- Etiquetas
    t.IS_FRAUD,
    t.IMPACTO_FRAUDE
FROM Transacciones t
JOIN Clientes c  ON c.id_cliente  = t.id_cliente
JOIN Cuentas cu  ON cu.id_cuenta  = t.id_cuenta
LEFT JOIN Tarjetas tk ON tk.id_tarjeta = t.id_tarjeta;
