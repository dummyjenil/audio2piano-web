import React, { useState } from 'react';
import { type detectionOptions } from "./toMidi";

type Props = {
    initialConfig: detectionOptions;
    onChange?: (updated: detectionOptions) => void;
};

const ConfigComponent: React.FC<Props> = ({ initialConfig, onChange }) => {
    const [config, setConfig] = useState<detectionOptions>(initialConfig);

    const handleChange = (
        key: keyof detectionOptions,
        value: string | number | boolean | null
    ) => {
        const newValue =
            value === '' ? undefined : typeof config[key] === 'number' ? Number(value) : value;

        const updatedConfig = {
            ...config,
            [key]: newValue,
        };

        setConfig(updatedConfig);
        if (onChange) {
            onChange(updatedConfig);
        }
    };

    return (
        <div>
            <h3>Detection Config</h3>

            {Object.entries(config).map(([key, value]) => (
                <div key={key} style={{ marginBottom: '10px' }}>
                    <label>
                        {key}:{' '}
                        {typeof value === 'boolean' ? (
                            <input
                                type="checkbox"
                                checked={value}
                                onChange={(e) => handleChange(key as keyof detectionOptions, e.target.checked)}
                            />
                        ) : (
                            <input
                                type="number"
                                value={value ?? ''}
                                onChange={(e) => handleChange(key as keyof detectionOptions, e.target.value)}
                            />
                        )}
                    </label>
                </div>
            ))}
        </div>
    );
};

export default ConfigComponent;
